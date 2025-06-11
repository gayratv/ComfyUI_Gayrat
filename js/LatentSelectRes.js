import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Model-specific size configurations
const MODEL_SIZES = {
    "SD": {
        "1:1": ["512x512", "640x640", "768x768"],
        "4:3": ["512x384", "640x480", "768x576"],
        "3:4": ["384x512", "480x640", "576x768"],
        "16:9": ["512x288", "640x360", "768x432", "896x512"],
        "9:16": ["288x512", "360x640", "432x768", "512x896"]
    },
    "SDXL": {
        "1:1": ["768x768", "896x896", "1024x1024", "1152x1152"],
        "4:3": ["768x576", "896x672", "1024x768", "1152x864"],
        "3:4": ["576x768", "672x896", "768x1024", "864x1152"],
        "16:9": ["768x432", "896x512", "1024x576", "1152x640", "1280x720"],
        "9:16": ["432x768", "512x896", "576x1024", "640x1152", "720x1280"]
    },
    "Flux": {
        // 896x896 - оптимальный размер для экономии памяти
        "1:1": ["896x896", "1024x1024", "1152x1152", "1280x1280", "1408x1408"],
        "4:3": ["896x672", "1024x768", "1152x864", "1280x960", "1408x1056"],
        "3:4": ["672x896", "768x1024", "864x1152", "960x1280", "1056x1408"],
        "16:9": ["896x512", "1024x576", "1152x640", "1280x720", "1408x800", "1536x864", "1920x1088"],
        "9:16": ["512x896", "576x1024", "640x1152", "720x1280", "800x1408", "864x1536", "1088x1920"]
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

                // Hide duplicate widgets
                const widgetsToCheck = ["control after generate", "control_after_generate"];
                let foundFirst = false;

                for (const widget of this.widgets) {
                    if (widgetsToCheck.includes(widget.name)) {
                        if (foundFirst) {
                            // Hide duplicate widget
                            widget.type = "hidden";
                            widget.computeSize = () => [0, -4];  // Minimize space
                        } else {
                            foundFirst = true;
                        }
                    }
                }

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

                // Force node size recalculation
                this.setSize(this.computeSize());

                return result;
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

            // Hide duplicate seed_control widget after load
            setTimeout(() => {
                const widgetsToCheck = ["control after generate", "control_after_generate", "seed control", "seed_control"];
                const foundWidgets = new Set();

                for (const widget of node.widgets) {
                    const normalizedName = widget.name.replace(/ /g, '_').toLowerCase();
                    if (foundWidgets.has(normalizedName)) {
                        // Hide duplicate widget
                        widget.type = "hidden";
                        widget.computeSize = () => [0, -4];
                    } else {
                        foundWidgets.add(normalizedName);
                    }
                }

                // Force node size recalculation
                node.setSize(node.computeSize());
            }, 100);
        }
    }
});