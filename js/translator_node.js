import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Лог 1: Проверяем, что файл вообще загружается
console.log("Debug: GoogleTranslateNode2 extension script loaded.");

app.registerExtension({
	name: "Comfy.GoogleTranslateNode2",
	nodeCreated(node) {
        // Лог 2: Проверяем, что ComfyUI видит создание узлов
		console.log("Debug: Node created on graph - ", node.title);

		// Проверяем, совпадает ли класс узла с нашим
		if (node.comfyClass === "GoogleTranslateNode2") {
            // Лог 3: Это самое важное сообщение. Если вы его видите, значит, связь по имени класса установлена!
			console.log("Debug: MATCH! Applying custom widget to", node.comfyClass);

			const widget = {
				type: "customtext",
				name: "translated_text_display",
				draw: function (ctx, node, width, y, height) {
					ctx.fillStyle = "#999";
					ctx.font = "12px Arial";
					ctx.textAlign = "left";
					const text = this.value || "[перевод появится здесь]"; // добавим текст по умолчанию
					const lines = text.split('\n');
					let lineY = y;
					for (const line of lines) {
						ctx.fillText(line, 10, lineY + 15);
						lineY += 20;
					}
				},
			};
			node.addCustomWidget(widget);
			const onExecuted = node.onExecuted;
			node.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);

                // Лог 4: Проверяем, доходит ли сообщение от бэкенда после выполнения
				console.log("Debug: Node executed. Message received:", message);

				if (message?.ui?.translated_text) {
                    // Лог 5: Проверяем, нашли ли мы нужные данные в сообщении
					console.log("Debug: UI data found! Text:", message.ui.translated_text[0]);

					const w = this.widgets.find((w) => w.name === "translated_text_display");
					if (w) {
						w.value = message.ui.translated_text[0];
						this.setDirty(true);
					}
				}
			};
		}
	},
});