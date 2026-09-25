/**
 * compact-thinking — densify old thinking blocks.
 *
 * The model keeps raw thinking for its last K actions; older blocks reach it
 * as short digests. A digest is made by a background call of the same model
 * over the same session prefix, so the provider reads it from the warm cache.
 *
 * Digests are substituted into the context on the fly by one function:
 * - in `context`, before the main model request;
 * - in `session_before_compact`, where Pi hands the same preparation object to
 *   its own summarizer after the hook.
 *
 * Nothing is written to the session: the raw text stays in the message entry,
 * while the state lives in extension entries. Pi therefore derives the context
 * size and the compaction threshold from the real `usage`, not from a rough
 * character estimate.
 *
 * Task notes: `~/.pi/agent/.task/current/19_compact-thinking/`. Extension
 * description — README.md, test scenarios — TESTING.md, pure rules — probe.mjs.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { getCurrentTools, type Message } from "@earendil-works/pi-ai";
import {
	convertToLlm,
	getAgentDir,
	type BeforeAgentStartEvent,
	type ContextEvent,
	type ContextEventResult,
	type ExtensionAPI,
	type ExtensionContext,
	type SessionBeforeCompactEvent,
	type SessionProjection,
	type SessionTreeEvent,
} from "@earendil-works/pi-coding-agent";
import {
	DEFAULT_CONFIG,
	blockKey,
	buildReplacement,
	foreignChangedBlocks,
	formatStatusText,
	formatTokens,
	isThinkingBlock,
	keepsSignature,
	parseConfig,
	selectCandidates,
	validateCompaction,
	type Candidate,
	type Config,
	type VisibleAssistant,
} from "./logic";

type AgentMessage = ContextEvent["messages"][number];
type AssistantContent = Extract<AgentMessage, { role: "assistant" }>["content"];

const STATUS_KEY = "compact-thinking";
/** Extension entry: the digest is substituted into the context. */
const APPLIED_ENTRY = "compact-thinking-applied";
/** Entries of the former extension name: read so old sessions keep their digests. */
const LEGACY_APPLIED_ENTRIES = ["hybrid-thinking-applied"];
/** Extension entry: the digest was rejected and is not retried. */
const REJECTED_ENTRY = "compact-thinking-rejected";
const LEGACY_REJECTED_ENTRIES = ["hybrid-thinking-rejected"];
/** How many transient failures one block survives per session. */
const MAX_ATTEMPTS = 3;
/** Status spinner frames: they take the place of the space while the job runs. */
const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const SPINNER_INTERVAL_MS = 100;

const COMPACTION_PROMPT = [
	"You are a compression tool, not the assistant in the transcript above.",
	"",
	"Rewrite the thinking block quoted below in the same language, shorter, cleaner",
	"and better organized. Keep every option considered and rejected with the",
	"reason it was rejected, every decision and its reason, every fact the",
	"reasoning relied on, every open question. Drop repetitions and filler.",
	"",
	"Wrap the result in <digest> and </digest> tags.",
].join("\n");

interface AppliedRecord {
	targetEntryId: string;
	blockIndex: number;
	rawTokens: number;
	compactedTokens: number;
	compactedText: string;
	spentTokens: number;
	spentOutput: number;
}

interface RejectedRecord {
	targetEntryId: string;
	blockIndex: number;
	reason: string;
	spentTokens: number;
	spentOutput: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

function asAppliedRecord(data: unknown): AppliedRecord | undefined {
	if (!isRecord(data)) return undefined;
	if (typeof data.targetEntryId !== "string" || typeof data.compactedText !== "string") return undefined;
	if (!Number.isFinite(data.blockIndex)) return undefined;
	return {
		targetEntryId: data.targetEntryId,
		blockIndex: Number(data.blockIndex),
		rawTokens: Number(data.rawTokens ?? 0),
		compactedTokens: Number(data.compactedTokens ?? 0),
		compactedText: data.compactedText,
		spentTokens: Number(data.spentTokens ?? 0),
		spentOutput: Number(data.spentOutput ?? 0),
	};
}

function asRejectedRecord(data: unknown): RejectedRecord | undefined {
	if (!isRecord(data)) return undefined;
	if (typeof data.targetEntryId !== "string" || !Number.isFinite(data.blockIndex)) return undefined;
	return {
		targetEntryId: data.targetEntryId,
		blockIndex: Number(data.blockIndex),
		reason: typeof data.reason === "string" ? data.reason : "rejected",
		spentTokens: Number(data.spentTokens ?? 0),
		spentOutput: Number(data.spentOutput ?? 0),
	};
}

function configPath(): string {
	return join(getAgentDir(), "compact-thinking.json");
}

function loadConfig(): { config: Config; problem?: string } {
	try {
		return { config: parseConfig(JSON.parse(readFileSync(configPath(), "utf8"))) };
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code === "ENOENT") {
			return { config: DEFAULT_CONFIG };
		}
		const message = error instanceof Error ? error.message : String(error);
		return { config: DEFAULT_CONFIG, problem: `config.json: ${message}` };
	}
}

function buildRequestText(blockText: string): string {
	return `${COMPACTION_PROMPT}\n\n<thinking_block>\n${blockText}\n</thinking_block>`;
}

/** Estimated like Pi does: four characters per token. */
const estimate = (text: string): number => Math.ceil(text.length / 4);

class HybridThinking {
	private readonly pi: ExtensionAPI;
	private readonly config: Config;
	private readonly configProblem: string | undefined;
	private readonly warned = new Set<string>();
	private readonly appliedRecords = new Map<string, AppliedRecord>();
	private readonly rejected = new Map<string, string>();
	private readonly attempts = new Map<string, number>();

	/**
	 * The background job must not touch `ctx`: it goes stale after session
	 * replacement or reload. References and values are captured while the event
	 * handler is still active.
	 */
	private manager: ExtensionContext["sessionManager"] | undefined;
	private registry: ExtensionContext["modelRegistry"] | undefined;
	private model: ExtensionContext["model"] | undefined;
	private ui: ExtensionContext["ui"] | undefined;
	private runPrompt = "";
	private systemOptions: { forceSystemPrompt?: string } | undefined;

	/** Context with digests substituted, keyed by assistant message time. */
	private substitutions = new Map<number, AssistantContent>();

	private enabled: boolean;
	private running = false;
	private spinnerIndex = 0;
	private spinner: ReturnType<typeof setInterval> | undefined;
	private job: AbortController | undefined;
	private generation = 0;
	private spentTokens = 0;
	private spentOutput = 0;
	private rejectedSpentTokens = 0;
	private rejectedSpentOutput = 0;
	private rawThinkingTokens = 0;
	private savedThinkingTokens = 0;

	constructor(pi: ExtensionAPI) {
		this.pi = pi;
		const loaded = loadConfig();
		this.config = loaded.config;
		this.configProblem = loaded.problem;
		this.enabled = loaded.config.enabled;

		pi.on("session_start", (_event, ctx) => this.onSessionStart(ctx));
		pi.on("session_tree", (_event: SessionTreeEvent, ctx) => this.onTree(ctx));
		pi.on("before_agent_start", (event: BeforeAgentStartEvent) => this.onBeforeAgentStart(event));
		pi.on("context", (event: ContextEvent) => this.onContext(event));
		pi.on("session_before_compact", (event: SessionBeforeCompactEvent) => this.onBeforeCompact(event));
		pi.on("tool_execution_start", (_event, ctx) => this.onToolExecutionStart(ctx));
		pi.on("turn_end", (_event, ctx) => this.onTurnEnd(ctx));
		pi.on("agent_before_settle", (_event, ctx) => this.onAgentBeforeSettle(ctx));
		pi.on("session_shutdown", () => this.stop());
		pi.registerCommand("compact-thinking", {
			description: "Densify old thinking blocks: report, on, off, dump",
			handler: async (args, ctx) => this.onCommand(args, ctx),
		});
	}

	private onSessionStart(ctx: ExtensionContext): void {
		this.capture(ctx);
		this.generation += 1;
		this.systemOptions = undefined;
		this.enabled = this.config.enabled;
		this.rebuild();
		if (this.configProblem) this.warn(`compact-thinking: ${this.configProblem}`);
		this.updateStatus();
	}

	/** Tree navigation does not emit `session_start`: the branch state is rebuilt. */
	private onTree(ctx: ExtensionContext): void {
		this.capture(ctx);
		this.rebuild();
		this.updateStatus();
	}

	/** Pi collapses a forced prompt (set by extensions like roles) into one head. */
	private onBeforeAgentStart(event: BeforeAgentStartEvent): void {
		this.systemOptions = event.systemPromptOptions;
	}

	/** Context before the model request: digests take the place of raw blocks. */
	private onContext(event: ContextEvent): ContextEventResult | undefined {
		const messages = this.substituteMessages(event.messages);
		return messages ? { messages } : undefined;
	}

	/** The same context for Pi's summarizer: the preparation is edited in place. */
	private onBeforeCompact(event: SessionBeforeCompactEvent): void {
		const summarized = this.substituteMessages(event.preparation.messagesToSummarize);
		if (summarized) event.preparation.messagesToSummarize = summarized;
		const prefix = this.substituteMessages(event.preparation.turnPrefixMessages);
		if (prefix) event.preparation.turnPrefixMessages = prefix;
	}

	/** The tool window: the background job does not compete with the model stream. */
	private onToolExecutionStart(ctx: ExtensionContext): void {
		this.capture(ctx);
		this.refreshThinkingTokens();
		setTimeout(() => void this.pump(), 0);
	}

	private onTurnEnd(ctx: ExtensionContext): void {
		this.capture(ctx);
		this.refreshThinkingTokens();
		this.updateStatus();
	}

	private onAgentBeforeSettle(ctx: ExtensionContext): void {
		this.capture(ctx);
		this.refreshThinkingTokens();
		this.updateStatus();
		setTimeout(() => void this.pump(), 0);
	}

	private onCommand(args: string, ctx: ExtensionContext): void {
		this.capture(ctx);
		const command = args.trim().toLowerCase();
		if (command === "on" || command === "off") {
			this.enabled = command === "on";
			this.updateStatus();
			this.notify(`compact-thinking: ${this.enabled ? "on" : "off"}`, "info");
			if (this.enabled) setTimeout(() => void this.pump(), 0);
			return;
		}
		if (command === "dump") {
			this.dump();
			return;
		}
		this.updateStatus();
		this.notify(this.reportText(), "info");
	}

	private capture(ctx: ExtensionContext): void {
		this.manager = ctx.sessionManager;
		this.registry = ctx.modelRegistry;
		this.model = ctx.model;
		this.ui = ctx.ui;
		this.runPrompt = ctx.getSystemPrompt();
	}

	/** Restore the branch state after restart, resume and navigation. */
	private rebuild(): void {
		const manager = this.manager;
		if (!manager) return;
		this.appliedRecords.clear();
		this.rejected.clear();
		this.attempts.clear();
		this.rejectedSpentTokens = 0;
		this.rejectedSpentOutput = 0;
		for (const entry of manager.getBranch()) {
			if (entry.type !== "custom") continue;
			if (entry.customType === APPLIED_ENTRY || LEGACY_APPLIED_ENTRIES.includes(entry.customType)) {
				const record = asAppliedRecord(entry.data);
				if (!record) continue;
				if (record.rawTokens < record.compactedTokens) continue;
				this.appliedRecords.set(blockKey(record.targetEntryId, record.blockIndex), record);
			} else if (entry.customType === REJECTED_ENTRY || LEGACY_REJECTED_ENTRIES.includes(entry.customType)) {
				const record = asRejectedRecord(entry.data);
				if (!record) continue;
				this.rejected.set(blockKey(record.targetEntryId, record.blockIndex), record.reason);
				this.rejectedSpentTokens += record.spentTokens;
				this.rejectedSpentOutput += record.spentOutput;
			}
		}
		this.rebuildSubstitutions();
	}

	/**
	 * Build the substituted context: for every target entry, its content with the
	 * replaced blocks. Counters take only active digests into account.
	 */
	private rebuildSubstitutions(): void {
		const manager = this.manager;
		this.substitutions = new Map();
		this.savedThinkingTokens = 0;
		this.spentTokens = this.rejectedSpentTokens;
		this.spentOutput = this.rejectedSpentOutput;
		if (!manager) return;
		const projection = manager.buildSessionProjection();
		const inContext = new Set(projection.entries.map((projected) => projected.sourceEntry.id));
		const byTarget = new Map<string, AppliedRecord[]>();
		for (const record of this.appliedRecords.values()) {
			if (!inContext.has(record.targetEntryId)) continue;
			const group = byTarget.get(record.targetEntryId) ?? [];
			group.push(record);
			byTarget.set(record.targetEntryId, group);
		}
		for (const [targetId, records] of byTarget) {
			const entry = manager.getEntry(targetId);
			if (entry?.type !== "message" || entry.message.role !== "assistant") continue;
			const ordered = [...records].sort((left, right) => left.blockIndex - right.blockIndex);
			let content = entry.message.content;
			for (const record of ordered) {
				const block = content[record.blockIndex];
				if (!block || block.type !== "thinking") continue;
				content = buildReplacement(
					content,
					record.blockIndex,
					record.compactedText,
					keepsSignature(block.thinkingSignature),
				);
			}
			this.substitutions.set(entry.message.timestamp, content);
			for (const record of ordered) {
				this.savedThinkingTokens += record.rawTokens - record.compactedTokens;
				this.spentTokens += record.spentTokens;
				this.spentOutput += record.spentOutput;
			}
		}
		this.rawThinkingTokens = this.measureRawThinking(projection);
	}

	/** Substitute digests into the messages; undefined when there is nothing to change. */
	private substituteMessages(messages: readonly AgentMessage[]): AgentMessage[] | undefined {
		if (this.substitutions.size === 0) return undefined;
		let changed = false;
		const next = messages.map((message) => {
			if (message.role !== "assistant") return message;
			const content = this.substitutions.get(message.timestamp);
			if (!content) return message;
			changed = true;
			return { ...message, content };
		});
		return changed ? next : undefined;
	}

	/**
	 * The oldest uncompacted block left of the cut point. A block whose visible
	 * text was changed by a foreign `context_edit` is left alone.
	 */
	private pickCandidate(): Candidate | undefined {
		const manager = this.manager;
		if (!manager) return undefined;
		const projection = manager.buildSessionProjection();
		const assistants: VisibleAssistant[] = [];
		const changed = new Set<string>();
		for (const projected of projection.entries) {
			const entry = projected.sourceEntry;
			if (entry.type !== "message" || entry.message.role !== "assistant") continue;
			const visible = projected.messages[0];
			if (!visible || visible.role !== "assistant" || !Array.isArray(visible.content)) continue;
			const content = entry.message.content;
			const foreign = foreignChangedBlocks(content, visible.content);
			if (foreign.length > 0) {
				content.forEach((block, index) => {
					if (block.type === "thinking") changed.add(blockKey(entry.id, index));
				});
			}
			assistants.push({ entryId: entry.id, api: entry.message.api, content });
		}
		const skip = new Set<string>([...this.appliedRecords.keys(), ...this.rejected.keys(), ...changed]);
		return selectCandidates(assistants, skip, this.config, estimate)[0];
	}

	/**
	 * The compaction call uses the same prefix as the main session. With a forced
	 * system prompt Pi collapses system messages into one head — mirrored here.
	 */
	private async compact(candidate: Candidate): Promise<void> {
		const manager = this.manager;
		const registry = this.registry;
		const model = this.model;
		if (!manager || !registry || !model) return;
		const generation = this.generation;
		const projection = manager.buildSessionProjection();
		const messages = convertToLlm(projection.messages);
		const tools = getCurrentTools(messages);
		const prefix: Message[] =
			this.systemOptions?.forceSystemPrompt !== undefined
				? [
						{
							role: "system",
							content: this.runPrompt,
							...(tools.length > 0 ? { toolsAdded: tools } : {}),
							timestamp: Date.now(),
						},
						...messages.filter((message) => message.role !== "system"),
					]
				: messages;
		prefix.push({
			role: "user",
			content: [{ type: "text", text: buildRequestText(candidate.text) }],
			timestamp: Date.now(),
		});

		const controller = new AbortController();
		this.job = controller;
		try {
			const stream = registry.streamSimple(model, { messages: prefix }, {
				maxTokens: Math.max(512, candidate.tokens),
				toolChoice: "none",
				sessionId: manager.getSessionId(),
				signal: controller.signal,
			});
			const iterator = stream[Symbol.asyncIterator]();
			while (!(await iterator.next()).done) {
				// Drain the stream: its event queue otherwise grows with the answer.
			}
			const message = await stream.result();
			if (generation !== this.generation || controller.signal.aborted) return;
			const spentTokens = message.usage?.totalTokens ?? 0;
			const spentOutput = message.usage?.output ?? 0;
			const verdict = validateCompaction(message, candidate.tokens, this.config.acceptanceRatio, estimate);
			const key = blockKey(candidate.entryId, candidate.blockIndex);
			if (!verdict.ok) {
				const reason = message.errorMessage ? `${verdict.reason}: ${message.errorMessage}` : verdict.reason;
				if (!verdict.terminal) {
					const attempts = (this.attempts.get(key) ?? 0) + 1;
					this.attempts.set(key, attempts);
					if (attempts < MAX_ATTEMPTS) {
						this.spentTokens += spentTokens;
						this.spentOutput += spentOutput;
						throw new Error(reason);
					}
				}
				this.attempts.delete(key);
				const record: RejectedRecord = {
					targetEntryId: candidate.entryId,
					blockIndex: candidate.blockIndex,
					reason,
					spentTokens,
					spentOutput,
				};
				this.persist(REJECTED_ENTRY, record);
				this.rejected.set(key, reason);
				this.rejectedSpentTokens += spentTokens;
				this.rejectedSpentOutput += spentOutput;
				return;
			}
			this.attempts.delete(key);
			const record: AppliedRecord = {
				targetEntryId: candidate.entryId,
				blockIndex: candidate.blockIndex,
				rawTokens: candidate.tokens,
				compactedTokens: verdict.tokens,
				compactedText: verdict.text,
				spentTokens,
				spentOutput,
			};
			this.persist(APPLIED_ENTRY, record);
			this.appliedRecords.set(key, record);
			this.rebuildSubstitutions();
		} finally {
			this.job = undefined;
		}
	}

	/** One background job: walk the candidates while there are any. */
	private async pump(): Promise<void> {
		if (this.running || !this.enabled) return;
		const generation = this.generation;
		this.running = true;
		this.refreshThinkingTokens();
		this.startSpinner();
		try {
			for (;;) {
				if (generation !== this.generation || !this.enabled) return;
				const candidate = this.pickCandidate();
				if (!candidate) return;
				await this.compact(candidate);
			}
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			this.warn(`compact-thinking: ${message}`);
		} finally {
			this.running = false;
			this.stopSpinner();
		}
	}

	private startSpinner(): void {
		if (this.spinner) return;
		this.spinnerIndex = 0;
		this.updateStatus();
		this.spinner = setInterval(() => {
			this.spinnerIndex = (this.spinnerIndex + 1) % SPINNER_FRAMES.length;
			this.updateStatus();
		}, SPINNER_INTERVAL_MS);
	}

	private stopSpinner(): void {
		if (this.spinner) clearInterval(this.spinner);
		this.spinner = undefined;
		this.updateStatus();
	}

	/**
	 * Next to the session file: `thinking-a` holds raw blocks, `thinking-b` what
	 * the model sees after substitution. Block headings match so a diff aligns.
	 */
	private dump(): void {
		const manager = this.manager;
		if (!manager) return;
		const sessionFile = manager.getSessionFile();
		if (!sessionFile) {
			this.notify("compact-thinking: the session is not persisted, nowhere to write", "warning");
			return;
		}
		const raw: string[] = [];
		const substituted: string[] = [];
		for (const projected of manager.buildSessionProjection().entries) {
			const entry = projected.sourceEntry;
			if (entry.type !== "message" || entry.message.role !== "assistant") continue;
			const content = this.substitutions.get(entry.message.timestamp);
			if (!content) continue;
			entry.message.content.forEach((block, index) => {
				if (block.type !== "thinking") return;
				if (!this.appliedRecords.has(blockKey(entry.id, index))) return;
				const replaced = content[index];
				const heading = `## ${entry.id}:${index} ${entry.timestamp}`;
				raw.push(`${heading}\n\n${block.thinking.trim()}`);
				substituted.push(`${heading}\n\n${isThinkingBlock(replaced) ? replaced.thinking.trim() : block.thinking.trim()}`);
			});
		}
		const dir = dirname(sessionFile);
		writeFileSync(join(dir, "thinking-a"), `${raw.join("\n\n")}\n`);
		writeFileSync(join(dir, "thinking-b"), `${substituted.join("\n\n")}\n`);
		this.notify(`compact-thinking: ${raw.length} digests in ${dir}`, "info");
	}

	private persist(customType: string, data: unknown): void {
		try {
			this.pi.appendEntry(customType, data);
		} catch {
			// The session is already replaced or closed: nowhere to put the entry.
		}
	}

	private statusText(): string {
		const frame = SPINNER_FRAMES[this.spinnerIndex] ?? " ";
		return formatStatusText(
			{
				enabled: this.enabled,
				running: this.running,
				rawThinkingTokens: this.rawThinkingTokens,
				contextThinkingTokens: this.rawThinkingTokens - this.savedThinkingTokens,
				generatedTokens: this.spentOutput,
			},
			frame,
		);
	}

	/** Detailed report for `/compact-thinking` without arguments. */
	private reportText(): string {
		return (
			`compact-thinking: ${this.enabled ? "on" : "off"} · ` +
			`digests in ${this.substitutions.size} messages · ` +
			`saved ${formatTokens(this.savedThinkingTokens)} tokens · ` +
			`generated ${formatTokens(this.spentOutput)} tokens · ` +
			`read ${formatTokens(this.spentTokens)} prefix tokens`
		);
	}

	/** Raw reasoning tokens in the current context. */
	private measureRawThinking(projection: SessionProjection): number {
		let raw = 0;
		for (const projected of projection.entries) {
			const entry = projected.sourceEntry;
			if (entry.type !== "message" || entry.message.role !== "assistant") continue;
			for (const block of entry.message.content) {
				if (block.type === "thinking") raw += estimate(block.thinking);
			}
		}
		return raw;
	}

	/** Recompute the raw reasoning sum for the current context. */
	private refreshThinkingTokens(): void {
		const manager = this.manager;
		if (!manager) return;
		this.rawThinkingTokens = this.measureRawThinking(manager.buildSessionProjection());
	}

	private updateStatus(): void {
		try {
			this.ui?.setStatus(STATUS_KEY, this.statusText());
		} catch {
			// Stale UI after a session replacement: the line no longer exists.
		}
	}

	private notify(message: string, level: "info" | "warning" | "error"): void {
		try {
			this.ui?.notify(message, level);
		} catch {
			// Stale UI after a session replacement.
		}
	}

	private warn(message: string): void {
		if (this.warned.has(message)) return;
		this.warned.add(message);
		this.notify(message, "warning");
	}

	private stop(): void {
		this.generation += 1;
		this.job?.abort();
		this.stopSpinner();
		this.running = false;
	}
}

export default function compactThinking(pi: ExtensionAPI): void {
	new HybridThinking(pi);
}
