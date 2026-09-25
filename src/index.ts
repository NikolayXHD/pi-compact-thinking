/**
 * compact-thinking — гибридный режим рассуждений.
 *
 * Последние K действий модель видит с сырыми thinking-блоками, более старые —
 * со сжатыми. Конспект делает фоновая компактификация той же моделью по
 * горячему префиксу сессии.
 *
 * Конспекты подставляются в контекст на лету, одной функцией:
 * - в `context` перед запросом основной модели;
 * - в `session_before_compact`, где pi после хука отдаёт свой preparation
 *   штатному сумматору.
 *
 * В сессию правки не пишутся: сырой текст остаётся в записи сообщения, а
 * состояние живёт в записях расширения. Поэтому pi считает контекст и порог
 * компактификации по реальному `usage`, без грубого пересчёта.
 *
 * Документация задачи: `~/.pi/agent/.task/current/19_compact-thinking/`.
 * Описание расширения — README.md, сценарии проверки — TESTING.md, прямой
 * прогон чистых правил — probe.mjs.
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
/** Запись расширения: конспект подставлен в контекст. */
const APPLIED_ENTRY = "compact-thinking-applied";
/** Записи прежнего имени расширения: читаем, чтобы конспекты старых сессий не терялись. */
const LEGACY_APPLIED_ENTRIES = ["hybrid-thinking-applied"];
/** Запись расширения: конспект отклонён, повтор не делается. */
const REJECTED_ENTRY = "compact-thinking-rejected";
const LEGACY_REJECTED_ENTRIES = ["hybrid-thinking-rejected"];
/** Сколько нетерминальных отказов терпит один блок за сессию. */
const MAX_ATTEMPTS = 3;
/** Кадры спиннера статуса: пробел после мозга на время работы. */
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
		reason: typeof data.reason === "string" ? data.reason : "отклонён",
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

/** Оценка как у pi: 4 символа на токен. */
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
	 * Фоновая задача не должна трогать `ctx`: он становится stale после
	 * замены или перезагрузки сессии. Ссылки и значения снимаются, пока
	 * обработчик события ещё активен.
	 */
	private manager: ExtensionContext["sessionManager"] | undefined;
	private registry: ExtensionContext["modelRegistry"] | undefined;
	private model: ExtensionContext["model"] | undefined;
	private ui: ExtensionContext["ui"] | undefined;
	private runPrompt = "";
	private systemOptions: { forceSystemPrompt?: string } | undefined;

	/** Контекст с подставленными конспектами, по времени assistant-сообщения. */
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
			description: "Гибридный режим рассуждений: состояние, on, off, dump",
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

	/** Переход по дереву не шлёт `session_start`: состояние ветки пересобирается. */
	private onTree(ctx: ExtensionContext): void {
		this.capture(ctx);
		this.rebuild();
		this.updateStatus();
	}

	/** Форсированный промпт (его ставит, например, roles) pi схлопывает в голову. */
	private onBeforeAgentStart(event: BeforeAgentStartEvent): void {
		this.systemOptions = event.systemPromptOptions;
	}

	/** Контекст перед запросом модели: конспекты встают на место сырых блоков. */
	private onContext(event: ContextEvent): ContextEventResult | undefined {
		const messages = this.substituteMessages(event.messages);
		return messages ? { messages } : undefined;
	}

	/** Тот же контекст для штатного сумматора pi: preparation правится на месте. */
	private onBeforeCompact(event: SessionBeforeCompactEvent): void {
		const summarized = this.substituteMessages(event.preparation.messagesToSummarize);
		if (summarized) event.preparation.messagesToSummarize = summarized;
		const prefix = this.substituteMessages(event.preparation.turnPrefixMessages);
		if (prefix) event.preparation.turnPrefixMessages = prefix;
	}

	/** Окно, когда агент занят инструментами: фон не спорит со стримингом модели. */
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
			this.notify(`compact-thinking: ${this.enabled ? "включён" : "выключен"}`, "info");
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

	/** Восстановить состояние ветки после перезапуска, возобновления и перехода. */
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
	 * Собрать контекст с конспектами: для каждой подстановки — содержимое
	 * записи с заменёнными блоками; счётчики считаются по активным конспектам.
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

	/** Подставить конспекты в сообщения; undefined, если менять нечего. */
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
	 * Самый старый несжатый блок левее точки отсечения. Блоки, чей видимый
	 * текст изменён чужим `context_edit`, не трогаем.
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
	 * Вызов компактификации по тому же префиксу, что у основной сессии.
	 * При форсированном system prompt pi схлопывает системные сообщения в
	 * одну голову — повторяем это.
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
				// Опустошаем поток: очередь событий иначе растёт вместе с ответом.
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

	/** Одна фоновая задача: идём по кандидатам, пока они есть. */
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
	 * Рядом с файлом сессии: `thinking-a` — сырые блоки, `thinking-b` — то,
	 * что видит модель после подстановки. Заголовки блоков совпадают, чтобы
	 * diff выравнивался.
	 */
	private dump(): void {
		const manager = this.manager;
		if (!manager) return;
		const sessionFile = manager.getSessionFile();
		if (!sessionFile) {
			this.notify("compact-thinking: сессия не сохраняется, складывать некуда", "warning");
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
		this.notify(`compact-thinking: ${raw.length} конспектов в ${dir}`, "info");
	}

	private persist(customType: string, data: unknown): void {
		try {
			this.pi.appendEntry(customType, data);
		} catch {
			// Сессия уже заменена или закрыта: запись некуда класть.
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

	/** Подробный отчёт для `/compact-thinking` без аргумента. */
	private reportText(): string {
		return (
			`compact-thinking: ${this.enabled ? "включён" : "выключен"} · ` +
			`конспектов на ${this.substitutions.size} сообщениях · ` +
			`сэкономлено ${formatTokens(this.savedThinkingTokens)} токенов · ` +
			`сгенерировано ${formatTokens(this.spentOutput)} токенов · ` +
			`прочитано префикса ${formatTokens(this.spentTokens)} токенов`
		);
	}

	/** Сырая сумма рассуждений в текущем контексте. */
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

	/** Пересчитать сырую сумму рассуждений по текущему контексту. */
	private refreshThinkingTokens(): void {
		const manager = this.manager;
		if (!manager) return;
		this.rawThinkingTokens = this.measureRawThinking(manager.buildSessionProjection());
	}

	private updateStatus(): void {
		try {
			this.ui?.setStatus(STATUS_KEY, this.statusText());
		} catch {
			// Stale UI после замены сессии: строка больше не существует.
		}
	}

	private notify(message: string, level: "info" | "warning" | "error"): void {
		try {
			this.ui?.notify(message, level);
		} catch {
			// Stale UI после замены сессии.
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
