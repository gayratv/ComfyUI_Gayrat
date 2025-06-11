import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Define the same data structure as in Python
const SIZES = {
    "1:1": [
        "512x512", "640x640", "768x768", "896x896",
        "1024x1024", "1152x1152", "1280x1280", "1408x1408"
    ],
    "4:3": [
        "512x384", "640x480", "768x576", "896x672",
        "1024x768", "1152x864", "1280x960", "1408x1056"
    ],
    "3:4": [
        "384x512", "480x640", "576x768", "672x896",
        "768x1024", "864x1152", "960x1280", "1056x1408"
    ],
    "16:9": [
        "512x288", "640x360", "768x432", "896x512",
        "1024x576", "1152x640", "1280x720", "1408x800",
        "1536x864", "1664x960", "1920x1088"
    ],
    "9:16": [
        "288x512", "360x640", "432x768", "512x896",
        "576x1024", "640x1152", "720x1280", "800x1408",
        "864x1536", "960x1664", "1088x1920"
    ]
};

// Model preferred sizes
const MODEL_PREFERRED_SIZES = {
    "Flux": "1024",
    "Flux PRO": "1024",
    "Flux Ultra": "1024",
    "SD": "512"
};

app.registerExtension({
    name: "Gayrat.FluxSDLatentImage",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "FluxSDLatentImage" || nodeData.name === "FluxSDLatentImageAdvanced") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);

                // Find the widgets
                const modelWidget = this.widgets.find(w => w.name === "model");
                const aspectRatioWidget = this.widgets.find(w => w.name === "aspect_ratio");
                const sizeWidget = this.widgets.find(w => w.name === "size");

                if (!modelWidget || !aspectRatioWidget || !sizeWidget) {
                    console.error("FluxSDLatentImage: Could not find required widgets");
                    return result;
                }

                // Function to update size dropdown
                const updateSizeOptions = () => {
                    const currentAspectRatio = aspectRatioWidget.value;
                    const currentModel = modelWidget.value;

                    // Get available sizes for current aspect ratio
                    const availableSizes = SIZES[currentAspectRatio] || SIZES["1:1"];

                    // Store current selection
                    const currentSize = sizeWidget.value;

                    // Update the options
                    sizeWidget.options.values = availableSizes;

                    // Determine best default size based on model
                    const preferredSize = MODEL_PREFERRED_SIZES[currentModel] || "1024";
                    let newSize = currentSize;

                    // If current size is not in new options, or if we should update based on model
                    if (!availableSizes.includes(currentSize)) {
                        // Find size containing preferred resolution
                        newSize = availableSizes.find(s => s.includes(preferredSize)) ||
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
        }
    },

    // Handle node serialization
    async nodeCreated(node, app) {
        if (node.comfyClass === "FluxSDLatentImage" || node.comfyClass === "FluxSDLatentImageAdvanced") {
            // Ensure size options are correct when loading from saved workflow
            const aspectRatioWidget = node.widgets?.find(w => w.name === "aspect_ratio");
            const sizeWidget = node.widgets?.find(w => w.name === "size");

            if (aspectRatioWidget && sizeWidget) {
                const availableSizes = SIZES[aspectRatioWidget.value] || SIZES["1:1"];
                sizeWidget.options.values = availableSizes;

                // Validate current value
                if (!availableSizes.includes(sizeWidget.value)) {
                    sizeWidget.value = availableSizes[0];
                }
            }
        }
    }
});