import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

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

// Default sizes for each model
const DEFAULT_SIZES = {
    "SD": "512x512",
    "SDXL": "1024x1024",
    "Flux": "896x896"  // Оптимально для экономии памяти
};

app.registerExtension({
    name: "Gayrat.FluxSDLatentImage",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "FluxSDLatentImage") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);

                // Remove duplicate widgets
                const removeDuplicateWidgets = () => {
                    const widgetNames = new Map();
                    const widgetsToRemove = [];

                    // Find all duplicate widgets
                    for (let i = 0; i < this.widgets.length; i++) {
                        const widget = this.widgets[i];
                        const normalizedName = widget.name.replace(/_/g, ' ').toLowerCase();

                        // Check if this is a duplicate of seed_control
                        if (normalizedName === "seed control" || normalizedName === "control after generate") {
                            if (widgetNames.has("seed_control")) {
                                widgetsToRemove.push(i);
                            } else {
                                widgetNames.set("seed_control", widget);
                            }
                        } else {
                            widgetNames.set(widget.name, widget);
                        }
                    }

                    // Remove duplicate widgets in reverse order to maintain indices
                    for (let i = widgetsToRemove.length - 1; i >= 0; i--) {
                        this.widgets.splice(widgetsToRemove[i], 1);
                    }
                };

                // Run immediately and after a delay to catch any delayed widget creation
                removeDuplicateWidgets();
                setTimeout(() => {
                    removeDuplicateWidgets();
                    this.setSize(this.computeSize());
                }, 10);

                // Find the widgets
                const modelWidget = this.widgets.find(w => w.name === "model");
                const aspectRatioWidget = this.widgets.find(w => w.name === "aspect_ratio");
                const sizeWidget = this.widgets.find(w => w.name === "size");

                if (!modelWidget || !aspectRatioWidget || !sizeWidget) {
                    console.error("FluxSDLatentImage: Could not find required widgets");
                    return result;
                }

                // Function to update size dropdown based on model and aspect ratio
                const updateSizeOptions = () => {
                    const currentModel = modelWidget.value;
                    const currentAspectRatio = aspectRatioWidget.value;

                    // Get available sizes for current model and aspect ratio
                    const modelSizes = MODEL_SIZES[currentModel] || MODEL_SIZES["Flux"];
                    const availableSizes = modelSizes[currentAspectRatio] || modelSizes["1:1"];

                    // Store current selection
                    const currentSize = sizeWidget.value;

                    // Update the options
                    sizeWidget.options.values = availableSizes;

                    // Determine new size
                    let newSize = currentSize;

                    // If current size is not in new options
                    if (!availableSizes.includes(currentSize)) {
                        // Try to find the default size for this model
                        const defaultSize = DEFAULT_SIZES[currentModel];

                        // Find default size or closest match
                        newSize = availableSizes.find(s => s === defaultSize) ||
                                 availableSizes.find(s => s.includes(defaultSize.split('x')[0])) ||
                                 availableSizes[Math.floor(availableSizes.length / 2)];
                    }

                    // Set the new value
                    sizeWidget.value = newSize;

                    // Force widget to update its display
                    if (sizeWidget.callback) {
                        sizeWidget.callback(newSize);
                    }
                };

                // Store original callbacks
                const originalModelCallback = modelWidget.callback;
                const originalAspectRatioCallback = aspectRatioWidget.callback;

                // Override model widget callback
                modelWidget.callback = function(value) {
                    if (originalModelCallback) {
                        originalModelCallback.call(this, value);
                    }
                    updateSizeOptions();
                };

                // Override aspect ratio widget callback
                aspectRatioWidget.callback = function(value) {
                    if (originalAspectRatioCallback) {
                        originalAspectRatioCallback.call(this, value);
                    }
                    updateSizeOptions();
                };

                // Initial setup
                updateSizeOptions();

                return result;
            };

            // Override addWidget to prevent duplicate creation
            const origAddWidget = nodeType.prototype.addWidget;
            nodeType.prototype.addWidget = function(type, name, value, callback, options) {
                // Check if we're trying to add a duplicate seed control widget
                const normalizedName = name.replace(/_/g, ' ').toLowerCase();
                if (normalizedName === "control after generate" || normalizedName === "seed control") {
                    // Check if we already have a seed control widget
                    const existingWidget = this.widgets.find(w =>
                        w.name === "seed_control" ||
                        w.name === "control after generate" ||
                        w.name.replace(/_/g, ' ').toLowerCase() === "seed control"
                    );
                    if (existingWidget) {
                        return existingWidget; // Return existing widget instead of creating new one
                    }
                }

                return origAddWidget.apply(this, arguments);
            };
        }
    },

    // Handle node serialization
    async nodeCreated(node, app) {
        if (node.comfyClass === "FluxSDLatentImage") {
            // Ensure size options are correct when loading from saved workflow
            const modelWidget = node.widgets?.find(w => w.name === "model");
            const aspectRatioWidget = node.widgets?.find(w => w.name === "aspect_ratio");
            const sizeWidget = node.widgets?.find(w => w.name === "size");

            if (modelWidget && aspectRatioWidget && sizeWidget) {
                const modelSizes = MODEL_SIZES[modelWidget.value] || MODEL_SIZES["Flux"];
                const availableSizes = modelSizes[aspectRatioWidget.value] || modelSizes["1:1"];
                sizeWidget.options.values = availableSizes;

                // Validate current value
                if (!availableSizes.includes(sizeWidget.value)) {
                    sizeWidget.value = availableSizes[0];
                }
            }
        }
    }
});