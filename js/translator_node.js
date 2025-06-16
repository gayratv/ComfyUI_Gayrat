import { app } from "../../../scripts/app.js";

/**
 * Вспомогательная функция для переноса текста по ширине.
 * @param {CanvasRenderingContext2D} ctx Контекст canvas для измерения текста.
 * @param {string} text Текст для переноса.
 * @param {number} maxWidth Максимальная ширина строки в пикселях.
 * @returns {string[]} Массив строк.
 */
function wrapText(ctx, text, maxWidth) {
    if (!text) return [];
	const words = text.split(' ');
	let lines = [];
	let currentLine = words[0];

	for (let i = 1; i < words.length; i++) {
		const word = words[i];
        // Проверяем, есть ли в слове перенос строки \n
        const splitWords = word.split('\n');
        if (splitWords.length > 1) {
            // Если есть, обрабатываем часть до переноса
            let testLine = currentLine + ' ' + splitWords[0];
            if (ctx.measureText(testLine).width > maxWidth) {
                lines.push(currentLine);
                currentLine = splitWords[0];
            } else {
                currentLine = testLine;
            }
            // Завершаем текущую строку и начинаем новые
            lines.push(currentLine);
            for(let j = 1; j < splitWords.length -1; j++) {
                lines.push(splitWords[j]);
            }
            currentLine = splitWords[splitWords.length - 1];
            continue;
        }

		const testLine = currentLine + ' ' + word;
		const metrics = ctx.measureText(testLine);
		const testWidth = metrics.width;
		if (testWidth > maxWidth && i > 0) {
			lines.push(currentLine);
			currentLine = word;
		} else {
			currentLine = testLine;
		}
	}
	lines.push(currentLine);
	return lines;
}


app.registerExtension({
	name: "Comfy.GoogleTranslateNode2.WithWrap",
	nodeCreated(node) {
		if (node.comfyClass === "GoogleTranslateNode2") {
			// Начальное значение
			node.translated_text = "[перевод появится здесь]";

			const widget = {
				type: "customtext",
				name: "translated_text_display",
				draw: function (ctx, node, width, y, height) {
					ctx.fillStyle = "#999";
					ctx.font = "12px Arial";
					ctx.textAlign = "left";

                    // Используем функцию для переноса текста
                    // Отступаем по 10px с каждой стороны для полей
					const lines = wrapText(ctx, node.translated_text, width - 20);

					let lineY = y + 15; // Начинаем рисовать с отступом
					for (const line of lines) {
                        if (lineY > y + node.size[1] - 10) break; // Не рисуем за пределами узла
						ctx.fillText(line, 10, lineY);
						lineY += 15; // Межстрочный интервал
					}
				},
			};

			node.addCustomWidget(widget);

			// Сохраняем оригинальную функцию, если она есть
			const onExecuted = node.onExecuted;
			node.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);

				if (message && message.translated_text) {
					const newText = message.translated_text[0];

					// --- ИСПРАВЛЕНИЕ ОШИБКИ ---
					// Используем 'node' вместо 'this'
					node.translated_text = newText;
					node.setDirty(true);
					app.graph.setDirtyCanvas(true, true);
				}
			};
		}
	},
});