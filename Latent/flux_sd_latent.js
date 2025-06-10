// ComfyUI extension for FluxSDLatentImage node
// Place this file in: ComfyUI/web/extensions/flux_sd_latent/

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "FluxSDLatent.DynamicSizes",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "FluxSDLatentImage") {
            // Store size options for each aspect ratio
            const sizeOptions = {
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

            // Model default sizes
            const modelDefaults = {
                "Flux": "1024x1024",
                "Flux PRO": "1024x1024",
                "Flux Ultra": "1024x1024",
                "SD": "512x512"
            };

            // Override onNodeCreated to add dynamic behavior
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);

                // Find the widgets
                const modelWidget = this.widgets.find(w => w.name === "model");
                const aspectWidget = this.widgets.find(w => w.name === "aspect_ratio");
                const sizeWidget = this.widgets.find(w => w.name === "size");

                if (modelWidget && aspectWidget && sizeWidget) {
                    // Function to update size options
                    const updateSizeOptions = () => {
                        const aspectRatio = aspectWidget.value;
                        const model = modelWidget.value;
                        const sizes = sizeOptions[aspectRatio] || sizeOptions["1:1"];

                        // Update the size widget options
                        sizeWidget.options.values = sizes;

                        // Set appropriate default based on model
                        const defaultSize = modelDefaults[model];
                        if (sizes.includes(defaultSize)) {
                            sizeWidget.value = defaultSize;
                        } else if (model.startsWith("Flux")) {
                            // For Flux, find closest to 1024x1024
                            sizeWidget.value = sizes.find(s => s.includes("1024")) || sizes[Math.floor(sizes.length / 2)];
                        } else {
                            // For SD, find closest to 512x512
                            sizeWidget.value = sizes.find(s => s.includes("512")) || sizes[0];
                        }

                        // Force widget update
                        if (sizeWidget.callback) {
                            sizeWidget.callback(sizeWidget.value);
                        }
                    };

                    // Add callbacks for when aspect ratio or model changes
                    const originalAspectCallback = aspectWidget.callback;
                    aspectWidget.callback = function(value) {
                        originalAspectCallback?.apply(this, arguments);
                        updateSizeOptions();
                    };

                    const originalModelCallback = modelWidget.callback;
                    modelWidget.callback = function(value) {
                        originalModelCallback?.apply(this, arguments);
                        updateSizeOptions();
                    };

                    // Initial update
                    updateSizeOptions();
                }

                return result;
            };

            // Add visual indicator for model type
            const onDrawBackground = nodeType.prototype.onDrawBackground;
            nodeType.prototype.onDrawBackground = function(ctx) {
                onDrawBackground?.apply(this, arguments);

                const modelWidget = this.widgets?.find(w => w.name === "model");
                if (modelWidget) {
                    // Draw model type indicator
                    ctx.save();
                    ctx.font = "12px Arial";
                    ctx.textAlign = "right";

                    if (modelWidget.value.startsWith("Flux")) {
                        ctx.fillStyle = "#4CAF50";
                        ctx.fillText("FLUX", this.size[0] - 10, 20);
                    } else {
                        ctx.fillStyle = "#2196F3";
                        ctx.fillText("SD", this.size[0] - 10, 20);
                    }

                    ctx.restore();
                }
            };
        }
    }
});