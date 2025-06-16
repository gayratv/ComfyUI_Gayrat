import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

const nodeName = "GoogleTranslateNode2";

app.registerExtension({
	name: "Gayrat.GoogleTranslateNode2.Final",

	/**
	 * Модифицирует прототип узла до его регистрации.
	 * Это гарантирует, что все экземпляры узла будут иметь нужную логику.
	 */
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === nodeName) {
			// Сохраняем оригинальную функцию onExecuted
			const onExecuted = nodeType.prototype.onExecuted;

			// Заменяем onExecuted на нашу собственную функцию
			nodeType.prototype.onExecuted = function(message) {
				// Вызываем оригинальную функцию, если она была
				onExecuted?.apply(this, arguments);

				// Проверяем, что в сообщении есть нужные данные
				if (message && message.text) {
					// Находим виджет по имени на данном экземпляре узла
					const widget = this.widgets.find((w) => w.name === "text_output");
					if (widget) {
						// Обновляем значение виджета
						widget.value = message.text[0];
					}
				}
			};
		}
	},

	/**
	 * Вызывается, когда узел создается на холсте.
	 * Здесь мы добавляем сам виджет.
	 */
	nodeCreated(node) {
		if (node.comfyClass === nodeName) {
			// Создаём стандартный многострочный виджет для вывода текста
			const widget = ComfyWidgets.STRING(node, "text_output", ["STRING", { multiline: true }], app);

			// Делаем виджет нередактируемым
			widget.widget.inputEl.readOnly = true;

			// Устанавливаем начальное значение
			widget.value = "[Перевод появится здесь]";
		}
	}
});