import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Добавляем определение узла в ComfyUI
app.registerExtension({
	name: "Gayrat.GoogleTranslateNode2",

	// Эта функция вызывается при создании узла на графе
	nodeCreated(node) {
		// Проверяем, является ли созданный узел нашим узлом "GoogleTranslateNode"
		if (node.comfyClass === "GoogleTranslateNode2") {

			// Создаем виджет для отображения переведенного текста
			const widget = {
				type: "customtext", // Уникальный тип виджета
				name: "translated_text_display", // Имя для доступа к виджету

				// Функция для отрисовки виджета
				draw: function (ctx, node, width, y, height) {
					// Настройки внешнего вида текста
					ctx.fillStyle = "#999"; // Цвет текста
					ctx.font = "12px Arial";
					ctx.textAlign = "left";

					// Получаем текст из значения виджета
					const text = this.value || "";
					const lines = text.split('\n');
					let lineY = y;

					// Рисуем каждую строку текста
					for (const line of lines) {
						ctx.fillText(line, 10, lineY + 15);
						lineY += 20; // Сдвигаем Y для следующей строки
					}
				},
			};

			// Добавляем наш новый виджет к узлу
			node.addCustomWidget(widget);

			// Сохраняем оригинальную функцию onExecuted
			const onExecuted = node.onExecuted;

			// Переопределяем функцию onExecuted для нашего узла
			node.onExecuted = function (message) {
				// Вызываем оригинальную функцию
				onExecuted?.apply(this, arguments);

				// Проверяем, есть ли в сообщении от бэкенда данные для нашего UI
				if (message?.ui?.translated_text) {
					// Находим наш виджет по имени
					const w = this.widgets.find((w) => w.name === "translated_text_display");
					if (w) {
						// Обновляем значение виджета
						w.value = message.ui.translated_text[0];
						// Запрашиваем перерисовку узла, чтобы отобразить изменения
						this.setDirty(true);
					}
				}
			};
		}
	},
});