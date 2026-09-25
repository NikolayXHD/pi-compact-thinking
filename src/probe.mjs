// Прямой прогон правил расширения через jiti: без сессии pi.
import { execFileSync } from "node:child_process";
import { realpathSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const piEntry = realpathSync(execFileSync("which", ["pi"], { encoding: "utf8" }).trim());
const piPackageDir = dirname(dirname(dirname(piEntry)));
const { createJiti } = await import(pathToFileURL(join(piPackageDir, "node_modules", "jiti", "lib", "jiti.mjs")).href);

const jiti = createJiti(import.meta.url);
const {
	parseConfig,
	DEFAULT_CONFIG,
	blockKey,
	canReplay,
	keepsSignature,
	selectCandidates,
	buildReplacement,
	foreignChangedBlocks,
	extractDigest,
	findToolProtocolMarker,
	validateCompaction,
	formatTokens,
	formatStatusText,
} = await jiti.import(fileURLToPath(new URL("./logic.ts", import.meta.url)));

let passed = 0;
let failed = 0;
const expectEqual = (name, got, want) => {
	if (JSON.stringify(got) === JSON.stringify(want)) {
		passed += 1;
		return;
	}
	failed += 1;
	console.log(`FAIL ${name}`);
	console.log(`  got:  ${JSON.stringify(got)}`);
	console.log(`  want: ${JSON.stringify(want)}`);
};
const expectTrue = (name, ok, detail) => {
	if (ok) {
		passed += 1;
		return;
	}
	failed += 1;
	console.log(`FAIL ${name}`);
	if (detail !== undefined) console.log(`  got: ${JSON.stringify(detail)}`);
};

const estimateText = (text) => Math.ceil(text.length / 4);
const thinkingBlock = (text, extra = {}) => ({ type: "thinking", thinking: text, ...extra });
const textBlock = (value) => ({ type: "text", text: value });
const repeatChar = (char, count) => char.repeat(count);
const candidateKeys = (candidates) => candidates.map((candidate) => blockKey(candidate.entryId, candidate.blockIndex));
const windowConfig = (k, minTokens) => ({ k, minTokens });
const attemptResult = (stopReason, content) => ({ stopReason, content });

expectEqual("пустой конфиг даёт значения по умолчанию", parseConfig(undefined), DEFAULT_CONFIG);
expectEqual(
	"поля конфига читаются",
	parseConfig({ enabled: false, k: 2, minTokens: 100, acceptanceRatio: 0.5 }),
	{ enabled: false, k: 2, minTokens: 100, acceptanceRatio: 0.5 },
);
expectEqual(
	"битые поля заменяются значениями по умолчанию",
	parseConfig({ k: -3, minTokens: "x", acceptanceRatio: 2 }),
	{ ...DEFAULT_CONFIG, k: 0 },
);
expectEqual("дробный k округляется вниз", parseConfig({ k: 2.9 }).k, 2);
expectEqual("строка вместо enabled не включает выключение", parseConfig({ enabled: "false" }).enabled, true);
expectEqual("доля приёмки ноль не принимается", parseConfig({ acceptanceRatio: 0 }).acceptanceRatio, DEFAULT_CONFIG.acceptanceRatio);
expectEqual("доля приёмки единица допустима", parseConfig({ acceptanceRatio: 1 }).acceptanceRatio, 1);

expectTrue("reasoning_content — подпись рассуждения", keepsSignature("reasoning_content"));
expectTrue("reasoning — подпись рассуждения", keepsSignature("reasoning"));
expectTrue("reasoning_text — подпись рассуждения", keepsSignature("reasoning_text"));
expectEqual("чужая подпись не reasoning-поле", keepsSignature("signed-by-server"), false);
expectEqual("подпись undefined не reasoning-поле", keepsSignature(undefined), false);

expectTrue(
	"подписанный блок openai-completions переотправляется",
	canReplay(thinkingBlock("мысль", { thinkingSignature: "reasoning_content" }), "openai-completions"),
);
expectEqual(
	"подпись не reasoning-полем — блок теряется",
	canReplay(thinkingBlock("мысль", { thinkingSignature: "sig" }), "openai-completions"),
	false,
);
expectEqual(
	"redacted не переотправляется",
	canReplay(thinkingBlock("мысль", { redacted: true, thinkingSignature: "reasoning_content" }), "openai-completions"),
	false,
);
expectEqual("redacted с чужой подписью не переотправляется", canReplay(thinkingBlock("мысль", { redacted: true, thinkingSignature: "sig" }), "openai-completions"), false);
expectEqual("пустой текст не переотправляется", canReplay(thinkingBlock("   "), "anthropic-messages"), false);
expectTrue("anthropic переотправляет блок без подписи", canReplay(thinkingBlock("мысль"), "anthropic-messages"));
expectTrue("неизвестный API переотправляет блок", canReplay(thinkingBlock("мысль"), "google-generative-ai"));

const longReasoning = repeatChar("a", 400);
const shortReasoning = repeatChar("b", 40);
const assistants = [
	{ entryId: "e1", api: "anthropic-messages", content: [thinkingBlock(longReasoning), textBlock("ok")] },
	{ entryId: "e2", api: "anthropic-messages", content: [thinkingBlock(longReasoning), { type: "toolCall" }, thinkingBlock(longReasoning)] },
	{ entryId: "e3", api: "anthropic-messages", content: [thinkingBlock(longReasoning)] },
];

expectEqual(
	"k держит последние блоки сырыми",
	candidateKeys(selectCandidates(assistants, new Set(), windowConfig(2, 1), estimateText)),
	["e1:0", "e2:0"],
);
expectEqual("k покрывает всю историю — кандидатов нет", candidateKeys(selectCandidates(assistants, new Set(), windowConfig(4, 1), estimateText)), []);
expectEqual("сжатые блоки пропускаются", candidateKeys(selectCandidates(assistants, new Set(["e1:0"]), windowConfig(2, 1), estimateText)), ["e2:0"]);
expectEqual("короткие блоки пропускаются", candidateKeys(selectCandidates(assistants, new Set(), windowConfig(2, 200), estimateText)), []);
expectEqual(
	"кандидаты идут от старых к новым",
	candidateKeys(selectCandidates(assistants, new Set(), windowConfig(1, 1), estimateText)),
	["e1:0", "e2:0", "e2:2"],
);
expectEqual(
	"текст блока обрезается по краям",
	selectCandidates([{ entryId: "e", api: "anthropic-messages", content: [thinkingBlock(`  ${longReasoning}  `)] }], new Set(), windowConfig(0, 1), estimateText)[0].text,
	longReasoning,
);
expectEqual(
	"redacted в сырой зоне не сдвигает отбор",
	candidateKeys(selectCandidates([
		{ entryId: "e1", api: "anthropic-messages", content: [thinkingBlock(longReasoning)] },
		{ entryId: "e2", api: "anthropic-messages", content: [thinkingBlock(longReasoning, { redacted: true })] },
	], new Set(), windowConfig(1, 1), estimateText)),
	["e1:0"],
);
expectEqual(
	"openai-completions без reasoning-подписи не сжимается",
	candidateKeys(selectCandidates([
		{ entryId: "e1", api: "openai-completions", content: [thinkingBlock(longReasoning, { thinkingSignature: "sig" })] },
		{ entryId: "e2", api: "openai-completions", content: [thinkingBlock(longReasoning, { thinkingSignature: "reasoning_content" })] },
	], new Set(), windowConfig(0, 1), estimateText)),
	["e2:0"],
);
expectEqual("пустая история — кандидатов нет", candidateKeys(selectCandidates([], new Set(), windowConfig(0, 1), estimateText)), []);
expectEqual(
	"k = 0 сжимает все блоки",
	candidateKeys(selectCandidates(assistants, new Set(), windowConfig(0, 1), estimateText)),
	["e1:0", "e2:0", "e2:2", "e3:0"],
);

const signatureBlock = thinkingBlock("старое", { thinkingSignature: "reasoning_content", redacted: false });
const responseBlock = textBlock("ответ");
const toolBlock = { type: "toolCall", id: "c1", name: "read", arguments: { path: "a" } };
const replaced = buildReplacement([signatureBlock, responseBlock, toolBlock], 0, "новое", true);
expectEqual("замена меняет только целевой блок", replaced[0].thinking, "новое");
expectEqual("подпись reasoning-поля сохраняется", replaced[0].thinkingSignature, "reasoning_content");
expectEqual("текст не тронут", replaced[1], responseBlock);
expectEqual("tool call не тронут", replaced[2], toolBlock);
expectEqual("чужая подпись снимается", buildReplacement([thinkingBlock("старое", { thinkingSignature: "sig" })], 0, "новое", false)[0].thinkingSignature, undefined);
expectEqual("подпись снята, текст заменён", buildReplacement([thinkingBlock("старое", { thinkingSignature: "sig" })], 0, "новое", false)[0].thinking, "новое");
expectEqual("индекс вне диапазона ничего не меняет", buildReplacement([signatureBlock, responseBlock], 7, "новое", false), [signatureBlock, responseBlock]);

const shownWithOwnEdit = [thinkingBlock("конспект", { thinkingSignature: "reasoning_content" }), thinkingBlock(longReasoning)];
expectEqual(
	"замена блока видна как расхождение с проекцией",
	foreignChangedBlocks([signatureBlock, thinkingBlock(longReasoning)], shownWithOwnEdit),
	[0],
);
expectEqual(
	"совпадающие блоки не считаются изменёнными",
	foreignChangedBlocks([signatureBlock, thinkingBlock(longReasoning)], [signatureBlock, thinkingBlock(longReasoning)]),
	[],
);
expectEqual("пропавший видимый блок — чужая правка", foreignChangedBlocks([thinkingBlock(longReasoning)], undefined), [0]);
expectEqual(
	"видимый блок другого типа — чужая правка",
	foreignChangedBlocks([thinkingBlock(longReasoning)], [textBlock("x")]),
	[0],
);

const lengthVerdict = validateCompaction(attemptResult("length", [textBlock(longReasoning)]), 100, 0.95, estimateText);
expectEqual("обрыв по длине не принимается", lengthVerdict.ok, false);
expectEqual("обрыв по длине — терминальный отказ", lengthVerdict.terminal, true);
expectEqual(
	"toolUse — терминальный отказ",
	validateCompaction(attemptResult("toolUse", [textBlock(longReasoning)]), 100, 0.95, estimateText).terminal,
	true,
);
const abortedVerdict = validateCompaction(attemptResult("aborted", []), 100, 0.95, estimateText);
expectEqual("обрыв запроса не принимается", abortedVerdict.ok, false);
expectEqual("обрыв запроса не терминален", abortedVerdict.terminal, false);
const errorVerdict = validateCompaction(attemptResult("error", []), 100, 0.95, estimateText);
expectEqual("ошибка провайдера не терминальна", errorVerdict.terminal, false);
expectEqual("ответ с tool call не принимается", validateCompaction(attemptResult("stop", [textBlock(longReasoning), toolBlock]), 100, 0.95, estimateText).terminal, true);
expectEqual("пустой ответ не принимается", validateCompaction(attemptResult("stop", [textBlock("  ")]), 100, 0.95, estimateText).ok, false);
expectEqual("выросший ответ не принимается", validateCompaction(attemptResult("stop", [textBlock(`<digest>${repeatChar("x", 400)}</digest>`)]), 100, 0.95, estimateText).ok, false);
expectEqual(
	"сокращённый ответ принимается",
	validateCompaction(attemptResult("stop", [textBlock(`<digest>${repeatChar("x", 200)}</digest>`)]), 100, 0.95, estimateText),
	{ ok: true, text: repeatChar("x", 200), tokens: 50 },
);
expectEqual(
	"доля приёмки ноль отвергает любой текст",
	validateCompaction(attemptResult("stop", [textBlock("коротко")]), 100, 0, estimateText).ok,
	false,
);
expectEqual(
	"ровно исходный размер на доле единица принимается",
	validateCompaction(attemptResult("stop", [textBlock(`<digest>${repeatChar("x", 400)}</digest>`)]), 100, 1, estimateText).tokens,
	100,
);
expectEqual("digest извлекается из обёртки", extractDigest("преамбула <digest>\nконспект\n</digest> хвост"), "конспект");
expectEqual("всё вне тегов игнорируется", extractDigest("мусор <digest>конспект</digest> мусор"), "конспект");
expectEqual("без тегов конспекта нет", extractDigest("просто текст"), undefined);
expectEqual("один тег не считается", extractDigest("<digest>конспект"), undefined);
expectEqual("закрытый тег раньше открытого не считается", extractDigest("</digest><digest>конспект"), undefined);
expectEqual(
	"текст конспекта склеивается и обрезается",
	validateCompaction(attemptResult("stop", [textBlock("  <digest> "), textBlock("первая\nвторая"), textBlock(" </digest>  ")]), 1000, 0.95, estimateText).text,
	"первая\nвторая",
);
expectEqual(
	"без обёртки ответ отклоняется",
	validateCompaction(attemptResult("stop", [textBlock(longReasoning)]), 100, 0.95, estimateText).ok,
	false,
);
expectEqual(
	"без обёртки отказ терминален",
	validateCompaction(attemptResult("stop", [textBlock("конспект")]), 100, 0.95, estimateText).terminal,
	true,
);
expectEqual(
	"пустая обёртка отклоняется",
	validateCompaction(attemptResult("stop", [textBlock("<digest>   </digest>")]), 100, 0.95, estimateText).ok,
	false,
);
expectEqual("чистый текст разметки не содержит", findToolProtocolMarker("обычное рассуждение"), undefined);
expectTrue(
	"DSML-разметка находится",
	findToolProtocolMarker("вывод ｜｜DSML｜｜ calls>") !== undefined,
);
expectTrue("markdown-тег tool_calls находится", findToolProtocolMarker("<tool_calls>") !== undefined);
expectEqual(
	"разметка внутри конспекта отклоняется",
	validateCompaction(attemptResult("stop", [textBlock(`<digest>${longReasoning}\n<tool_calls></digest>`)]), 100, 0.95, estimateText).ok,
	false,
);
expectEqual(
	"разметка вне конспекта не мешает",
	validateCompaction(attemptResult("stop", [textBlock("<tool_calls><digest>конспект</digest>")]), 100, 0.95, estimateText).ok,
	true,
);
expectEqual(
	"отказ из-за разметки терминален",
	validateCompaction(attemptResult("stop", [textBlock("<digest><｜｜DSML｜｜ invoke></digest>")]), 100, 0.95, estimateText).terminal,
	true,
);

expectEqual("токены до тысячи пишутся числом", formatTokens(999), "999");
expectEqual("тысячи округляются до целых", formatTokens(1500), "2k");
expectEqual("округление вниз", formatTokens(13400), "13k");
expectEqual("округление вверх", formatTokens(13500), "14k");
expectEqual("десятки тысяч", formatTokens(150000), "150k");
expectEqual("пустой контекст", formatTokens(0), "0");
expectEqual(
	"выключенный режим виден",
	formatStatusText({ enabled: false, running: false, rawThinkingTokens: 0, contextThinkingTokens: 0, generatedTokens: 0 }),
	"💭 off",
);
expectEqual(
	"статус показывает экономию, сырую сумму и генерацию",
	formatStatusText({ enabled: true, running: false, rawThinkingTokens: 120000, contextThinkingTokens: 80000, generatedTokens: 76000 }),
	"💭 -40k/120k +76k",
);
expectEqual(
	"во время работы пробел занимает кадр спиннера",
	formatStatusText({ enabled: true, running: true, rawThinkingTokens: 120000, contextThinkingTokens: 80000, generatedTokens: 76000 }, "⠹"),
	"💭⠹-40k/120k +76k",
);
expectEqual(
	"без сжатий экономия нулевая",
	formatStatusText({ enabled: true, running: false, rawThinkingTokens: 5000, contextThinkingTokens: 5000, generatedTokens: 0 }),
	"💭 0/5k +0",
);
expectEqual(
	"пустой контекст",
	formatStatusText({ enabled: true, running: false, rawThinkingTokens: 0, contextThinkingTokens: 0, generatedTokens: 0 }),
	"💭 0/0 +0",
);

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exitCode = 1;
