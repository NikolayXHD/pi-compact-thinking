/**
 * compact-thinking — pure rules for compacting old thinking blocks.
 *
 * The module is pure: no files, no Pi events. Everything runtime-bound lives
 * in index.ts. Test scenarios — TESTING.md, direct run — probe.mjs.
 */

export interface Config {
	enabled: boolean;
	/** How many last actions keep raw thinking blocks. */
	k: number;
	/** Blocks shorter than this are not compacted. */
	minTokens: number;
	/** A digest longer than this fraction of the source block is rejected. */
	acceptanceRatio: number;
}

export const DEFAULT_CONFIG: Config = {
	enabled: true,
	k: 5,
	minTokens: 512,
	acceptanceRatio: 0.95,
};

export function parseConfig(raw: unknown): Config {
	const source = typeof raw === "object" && raw !== null ? (raw as Record<string, unknown>) : {};
	const number = (value: unknown, fallback: number): number =>
		typeof value === "number" && Number.isFinite(value) ? value : fallback;
	const ratio = number(source.acceptanceRatio, DEFAULT_CONFIG.acceptanceRatio);
	return {
		enabled: typeof source.enabled === "boolean" ? source.enabled : DEFAULT_CONFIG.enabled,
		k: Math.max(0, Math.floor(number(source.k, DEFAULT_CONFIG.k))),
		minTokens: Math.max(0, Math.floor(number(source.minTokens, DEFAULT_CONFIG.minTokens))),
		acceptanceRatio: ratio > 0 && ratio <= 1 ? ratio : DEFAULT_CONFIG.acceptanceRatio,
	};
}

export interface ThinkingBlock {
	type: "thinking";
	thinking: string;
	thinkingSignature?: string;
	redacted?: boolean;
}

export interface TextBlock {
	type: "text";
	text: string;
}

export interface ToolCallBlock {
	type: "toolCall";
}

export type ContentBlock = ThinkingBlock | TextBlock | ToolCallBlock | { type: string };

/** Signatures with which a provider marks replayable reasoning. */
const REASONING_SIGNATURES = new Set(["reasoning_content", "reasoning", "reasoning_text"]);

export function isThinkingBlock(block: ContentBlock): block is ThinkingBlock {
	return block.type === "thinking" && "thinking" in block && typeof block.thinking === "string";
}

export function keepsSignature(signature: unknown): boolean {
	return typeof signature === "string" && REASONING_SIGNATURES.has(signature);
}

/**
 * Whether the provider replays this block's reasoning in the next request.
 * `openai-completions` sends reasoning only with a field-name signature; without
 * one the block is silently dropped, so there is nothing to compact.
 */
export function canReplay(block: ThinkingBlock, api: string): boolean {
	if (block.redacted || block.thinking.trim() === "") return false;
	if (api === "openai-completions") return keepsSignature(block.thinkingSignature);
	return true;
}

export function blockKey(entryId: string, blockIndex: number): string {
	return `${entryId}:${blockIndex}`;
}

/**
 * Indices of thinking blocks whose visible text was changed by a foreign
 * `context_edit`: our digests are not visible in the projection, so any
 * difference is foreign.
 */
export function foreignChangedBlocks(
	content: readonly ContentBlock[],
	shown: readonly ContentBlock[] | undefined,
): number[] {
	const changed: number[] = [];
	content.forEach((block, index) => {
		if (!isThinkingBlock(block)) return;
		const shownBlock = shown?.[index];
		if (!shownBlock || !isThinkingBlock(shownBlock) || shownBlock.thinking !== block.thinking) {
			changed.push(index);
		}
	});
	return changed;
}

export interface VisibleAssistant {
	entryId: string;
	api: string;
	content: readonly ContentBlock[];
}

export interface Candidate {
	entryId: string;
	blockIndex: number;
	text: string;
	tokens: number;
}

/**
 * Candidates are thinking blocks left of the cut point: the last K blocks stay
 * raw, everything older is compacted. Returned in order of appearance.
 */
export function selectCandidates(
	assistants: readonly VisibleAssistant[],
	skip: ReadonlySet<string>,
	config: Pick<Config, "k" | "minTokens">,
	estimate: (text: string) => number,
): Candidate[] {
	const blocks: { entryId: string; blockIndex: number; block: ThinkingBlock; api: string }[] = [];
	for (const assistant of assistants) {
		assistant.content.forEach((block, blockIndex) => {
			if (!isThinkingBlock(block)) return;
			blocks.push({ entryId: assistant.entryId, blockIndex, block, api: assistant.api });
		});
	}
	const cutoff = Math.max(0, blocks.length - config.k);
	const candidates: Candidate[] = [];
	for (const item of blocks.slice(0, cutoff)) {
		if (skip.has(blockKey(item.entryId, item.blockIndex))) continue;
		if (!canReplay(item.block, item.api)) continue;
		const text = item.block.thinking.trim();
		const tokens = estimate(text);
		if (tokens < config.minTokens) continue;
		candidates.push({ entryId: item.entryId, blockIndex: item.blockIndex, text, tokens });
	}
	return candidates;
}

/**
 * A one-to-one replacement: only the reasoning text of the target block
 * changes; answer text, tool calls and their signatures are untouched. A
 * signature that does not name a reasoning field is dropped: after a rewrite
 * such a signature is invalid.
 */
export function buildReplacement<T extends ContentBlock>(
	content: readonly T[],
	blockIndex: number,
	compactedText: string,
	keepSignature: boolean,
): T[] {
	return content.map((block, index) => {
		if (index !== blockIndex) return block;
		const replacement: Record<string, unknown> = { ...block, thinking: compactedText };
		if (!keepSignature) delete replacement.thinkingSignature;
		return replacement as unknown as T;
	});
}

export interface AttemptResult {
	stopReason: string;
	content: readonly ContentBlock[];
}

/**
 * Tool-call markup that leaked into text: providers return such attempts as
 * text when no tools are declared.
 */
const TOOL_PROTOCOL_PATTERNS = [
	/｜｜DSML/,
	/<｜tool_calls?｜>/,
	/<tool_calls?>/,
	/<invoke\s+name=/,
	/<function_calls?>/,
	/<antml:invoke/,
];

export function findToolProtocolMarker(text: string): string | undefined {
	return TOOL_PROTOCOL_PATTERNS.find((pattern) => pattern.test(text))?.source;
}

const DIGEST_OPEN = /<digest>/i;
const DIGEST_CLOSE = /<\/digest>/i;

/**
 * A digest must be wrapped in `<digest>…</digest>`: the contract defines the
 * result boundary, and anything the model wrote outside the tags (preamble,
 * transcript continuation, a tool call) never enters the digest.
 */
export function extractDigest(text: string): string | undefined {
	const open = DIGEST_OPEN.exec(text);
	if (!open) return undefined;
	const close = DIGEST_CLOSE.exec(text.slice(open.index + open[0].length));
	if (!close) return undefined;
	const start = open.index + open[0].length;
	return text.slice(start, start + close.index).trim();
}

export type Verdict =
	| { ok: true; text: string; tokens: number }
	| { ok: false; reason: string; terminal: boolean };

export function validateCompaction(
	result: AttemptResult,
	rawTokens: number,
	acceptanceRatio: number,
	estimate: (text: string) => number,
): Verdict {
	if (result.stopReason !== "stop") {
		return {
			reason: `stop reason: ${result.stopReason}`,
			terminal: result.stopReason !== "error" && result.stopReason !== "aborted",
			ok: false,
		};
	}
	if (result.content.some((block) => block.type === "toolCall")) {
		return { ok: false, reason: "the answer contains a tool call", terminal: true };
	}
	const text = result.content
		.filter((block) => block.type === "text" && "text" in block && typeof block.text === "string")
		.map((block) => (block as { text: string }).text.trim())
		.filter((part) => part !== "")
		.join("\n");
	const digest = extractDigest(text);
	if (digest === undefined) {
		return { ok: false, reason: "no <digest>…</digest> wrapper", terminal: true };
	}
	if (digest === "") {
		return { ok: false, reason: "empty digest", terminal: true };
	}
	const marker = findToolProtocolMarker(digest);
	if (marker) {
		return { ok: false, reason: `tool-call markup in the digest: ${marker}`, terminal: true };
	}
	const tokens = estimate(digest);
	if (tokens > rawTokens * acceptanceRatio) {
		return {
			ok: false,
			reason: `not shorter: ${tokens} > ${Math.floor(rawTokens * acceptanceRatio)}`,
			terminal: true,
		};
	}
	return { ok: true, text: digest, tokens };
}

export function formatTokens(tokens: number): string {
	if (tokens < 1000) return String(tokens);
	return `${Math.round(tokens / 1000)}k`;
}

export interface StatusCounts {
	enabled: boolean;
	running: boolean;
	/** Raw reasoning tokens in the context. */
	rawThinkingTokens: number;
	/** Reasoning tokens the model currently sees. */
	contextThinkingTokens: number;
	/** Tokens generated by the background calls. */
	generatedTokens: number;
}

export function formatStatusText(counts: StatusCounts, spinnerFrame = " "): string {
	if (!counts.enabled) return "💭 off";
	const separator = counts.running ? spinnerFrame : " ";
	const saved = counts.rawThinkingTokens - counts.contextThinkingTokens;
	const savedText = saved > 0 ? `-${formatTokens(saved)}` : "0";
	return `💭${separator}${savedText}/${formatTokens(counts.rawThinkingTokens)} +${formatTokens(counts.generatedTokens)}`;
}
