/**
 * compact-thinking — правила сжатия старых thinking-блоков.
 *
 * Модуль чистый: ни файлов, ни событий pi. Всё, что зависит от runtime,
 * живёт в index.ts. Сценарии проверки — TESTING.md, прямой прогон —
 * probe.mjs.
 */

export interface Config {
	enabled: boolean;
	/** Сколько последних действий остаются с сырыми thinking-блоками. */
	k: number;
	/** Блоки короче этого числа токенов не сжимаются. */
	minTokens: number;
	/** Конспект длиннее этой доли исходного блока не принимается. */
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

/** Подписи, которыми провайдер помечает переотправляемое рассуждение. */
const REASONING_SIGNATURES = new Set(["reasoning_content", "reasoning", "reasoning_text"]);

export function isThinkingBlock(block: ContentBlock): block is ThinkingBlock {
	return block.type === "thinking" && "thinking" in block && typeof block.thinking === "string";
}

export function keepsSignature(signature: unknown): boolean {
	return typeof signature === "string" && REASONING_SIGNATURES.has(signature);
}

/**
 * Переотправляет ли провайдер рассуждение такого блока в следующий запрос.
 * `openai-completions` шлёт рассуждение только при подписи-поле, без неё
 * блок молча теряется, и сжимать его незачем.
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
 * Индексы thinking-блоков, чей видимый текст изменён чужим `context_edit`: в
 * проекции наши конспекты не видны, поэтому любое расхождение — чужое.
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
 * Кандидаты — thinking-блоки левее точки отсечения: последние K блоков
 * остаются сырыми, всё старше сжимается. Возвращаются в порядке появления.
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
 * Замена 1 к 1: у целевого блока меняется только текст рассуждения, текст,
 * tool calls и их подписи не трогаются. Подпись снимается, если она не
 * называет reasoning-поле: после перезаписи такая подпись невалидна.
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
 * Разметка вызова инструмента, попавшая в текст: провайдер возвращает такие
 * попытки текстом, когда инструменты не объявлены.
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
 * Конспект обязан быть обёрнут в `<digest>…</digest>`: граница результата
 * задаётся контрактом, а всё, что модель написала за тегами (преамбула,
 * продолжение транскрипта, вызов инструмента), в конспект не идёт.
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
		return { ok: false, reason: "ответ содержит tool call", terminal: true };
	}
	const text = result.content
		.filter((block) => block.type === "text" && "text" in block && typeof block.text === "string")
		.map((block) => (block as { text: string }).text.trim())
		.filter((part) => part !== "")
		.join("\n");
	const digest = extractDigest(text);
	if (digest === undefined) {
		return { ok: false, reason: "нет обёртки <digest>…</digest>", terminal: true };
	}
	if (digest === "") {
		return { ok: false, reason: "пустой конспект", terminal: true };
	}
	const marker = findToolProtocolMarker(digest);
	if (marker) {
		return { ok: false, reason: `в конспекте разметка вызова инструмента: ${marker}`, terminal: true };
	}
	const tokens = estimate(digest);
	if (tokens > rawTokens * acceptanceRatio) {
		return {
			ok: false,
			reason: `не короче: ${tokens} > ${Math.floor(rawTokens * acceptanceRatio)}`,
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
	/** Сумма токенов сырых рассуждений в контексте. */
	rawThinkingTokens: number;
	/** Сумма токенов рассуждений, которые сейчас видит модель. */
	contextThinkingTokens: number;
	/** Сколько токенов сгенерировали фоновые вызовы. */
	generatedTokens: number;
}

export function formatStatusText(counts: StatusCounts, spinnerFrame = " "): string {
	if (!counts.enabled) return "💭 off";
	const separator = counts.running ? spinnerFrame : " ";
	const saved = counts.rawThinkingTokens - counts.contextThinkingTokens;
	const savedText = saved > 0 ? `-${formatTokens(saved)}` : "0";
	return `💭${separator}${savedText}/${formatTokens(counts.rawThinkingTokens)} +${formatTokens(counts.generatedTokens)}`;
}
