import { app } from "../../../scripts/app.js";
import { api } from '../../../scripts/api.js';

// Gayrat: кастомный виджет для выбора предустановленных разрешений
function showImage(name) {
    const img = new Image();
    img.src = name;
    const parent_element = document.getElementById("gayrat-comfyui-preview-image-div");
    parent_element.innerHTML = "";
    parent_element.appendChild(img);
}

function get_size_by_name(name) {
    if (!name) {
        return null;
    }
    const size = name.split("x");
    return [parseInt(size[0]), parseInt(size[1])];
}

// -- Словарь с разрешениями, который должен совпадать с Python-кодом
const MODEL_SIZES = {
	"SD": {
        "1:1": ["512x512", "640x640", "768x768"],
        "4:3": ["512x384", "768x576", "1024x768"],
        "3:2": ["512x344", "768x512", "960x640"],
        "16:9": ["512x288", "768x432", "1024x576"],
        "21:9": ["768x320", "1024x432"],
        "3:4": ["384x512", "576x768", "768x1024"],
        "2:3": ["344x512", "512x768", "640x960"],
        "9:16": ["288x512", "432x768", "576x1024"],
        "9:21": ["320x768", "432x1024"],
	},
	"SDXL": {
		"1:1": ["1024x1024", "896x896", "768x768", "512x512"],
        "4:3": ["1152x896", "1024x768"],
        "3:2": ["1216x832", "1024x688", "896x592"],
        "16:9": ["1344x768", "1216x688", "1024x576"],
        "21:9": ["1536x640", "1280x544"],
        "3:4": ["896x1152", "768x1024"],
        "2:3": ["832x1216", "688x1024", "592x896"],
        "9:16": ["768x1344", "688x1216", "576x1024"],
        "9:21": ["640x1536", "544x1280"]
	},
    "Flux": {
        "1:1": ["1024x1024", "896x896", "768x768"],
        "4:3": ["1152x896", "1024x768"],
        "3:2": ["1216x832", "1024x688"],
        "16:9": ["1344x768", "1024x576"],
        "3:4": ["896x1152", "768x1024"],
        "2:3": ["832x1216", "688x1024"],
        "9:16": ["768x1344", "576x1024"],
    },
    // Для SD3 используются те же разрешения, что и для SDXL
    "SD3": {
		"1:1": ["1024x1024", "896x896", "768x768", "512x512"],
        "4:3": ["1152x896", "1024x768"],
        "3:2": ["1216x832", "1024x688", "896x592"],
        "16:9": ["1344x768", "1216x688", "1024x576"],
        "21:9": ["1536x640", "1280x544"],
        "3:4": ["896x1152", "768x1024"],
        "2:3": ["832x1216", "688x1024", "592x896"],
        "9:16": ["768x1344", "688x1216", "576x1024"],
        "9:21": ["640x1536", "544x1280"]
	}
};

function updateWidget(node, widget_name, widget_value) {
    if (!node.widgets) {
        return;
    }

    const widget = node.widgets.find(w => w.name === widget_name);
    if (widget) {
        widget.value = widget_value;
        if (widget.callback) {
            widget.callback(widget.value, app.canvas, node, app.graph);
        }
    }

    if (widget_name === "model" || widget_name === "aspect_ratio") {
        const model = node.widgets.find(w => w.name === "model").value;
        const aspect_ratio = node.widgets.find(w => w.name === "aspect_ratio").value;
        const size_widget = node.widgets.find(w => w.name === "size");

        let default_size;
        if (model === 'Flux') {
            default_size = "896x896";
        } else if (model === 'SD' || model === 'SDXL' || model === 'SD3') {
            default_size = "1024x1024";
        }

        const sizes = MODEL_SIZES[model][aspect_ratio] || [default_size];
        size_widget.options.values = sizes;

        if (!sizes.includes(size_widget.value)) {
            size_widget.value = sizes[0];
            if (size_widget.callback) {
                size_widget.callback(size_widget.value, app.canvas, node, app.graph);
            }
        }
    }
}

app.registerExtension({
	name: "gayrat.LatentSelectRes",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === 'FluxSDLatentImage') {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Предотвращение дублирования виджетов
                if (this.widgets && this.widgets.some(w => w.name === 'aspect_ratio')) {
                    return;
                }

				const aspect_ratio_widget = this.addWidget("combo", "aspect_ratio", "1:1", (value) => {
                    updateWidget(this, "aspect_ratio", value);
                }, {
                    values: Object.keys(MODEL_SIZES["Flux"])
                });

				const size_widget = this.addWidget("combo", "size", "1024x1024", (value) => {
					const wh = get_size_by_name(value);
					if (wh) {
						updateWidget(this, "width", wh[0]);
						updateWidget(this, "height", wh[1]);
					}
				}, {
                    values: MODEL_SIZES["Flux"]["1:1"]
                });

                // Привязка обратных вызовов для виджетов
                this.widgets.find(w => w.name === 'model').callback = (value) => {
                    updateWidget(this, 'model', value);
                };

                // Начальная инициализация
                updateWidget(this, "model", this.widgets.find(w => w.name === 'model').value);
			};
		}
	},
});