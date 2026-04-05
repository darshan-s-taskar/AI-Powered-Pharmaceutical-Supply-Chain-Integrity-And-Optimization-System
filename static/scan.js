(function () {
    const state = window.__SCAN_PAGE__ || { latestResult: null };

    const uploadForm = document.getElementById("uploadForm");
    const submitImageButton = document.getElementById("submitImageButton");
    const feedbackEl = document.getElementById("scanFeedback");
    const processingIndicator = document.getElementById("processingIndicator");
    const resultEmptyState = document.getElementById("resultEmptyState");
    const resultContent = document.getElementById("resultContent");
    const resultLabel = document.getElementById("resultLabel");
    const resultConfidence = document.getElementById("resultConfidence");
    const resultImage = document.getElementById("resultImage");
    const ocrText = document.getElementById("ocrText");
    const resnetScore = document.getElementById("resnetScore");
    const efficientnetScore = document.getElementById("efficientnetScore");
    const vitScore = document.getElementById("vitScore");
    const ocrScore = document.getElementById("ocrScore");
    const ensembleScore = document.getElementById("ensembleScore");
    const historySavedChip = document.getElementById("historySavedChip");
    const verdictCard = document.getElementById("verdictCard");
    const cameraFeed = document.getElementById("cameraFeed");
    const cameraCanvas = document.getElementById("cameraCanvas");
    const startCameraButton = document.getElementById("startCameraButton");
    const captureButton = document.getElementById("captureButton");
    const barcodeValue = document.getElementById("barcodeValue");
    const barcodeStatus = document.getElementById("barcodeStatus");
    const startBarcodeButton = document.getElementById("startBarcodeButton");
    const stopBarcodeButton = document.getElementById("stopBarcodeButton");
    let cameraStream = null;
    let barcodeScanner = null;
    let barcodeActive = false;
    let isProcessing = false;

    function setFeedback(message, type) {
        feedbackEl.textContent = message;
        feedbackEl.className = "feedback";
        if (type) {
            feedbackEl.classList.add(type);
        }
    }

    function setProcessingState(active) {
        isProcessing = active;
        processingIndicator.classList.toggle("hidden", !active);
        submitImageButton.disabled = active;
        captureButton.disabled = active;
        startCameraButton.disabled = active;
        startBarcodeButton.disabled = active;
        stopBarcodeButton.disabled = active && !barcodeActive;
    }

    function formatPercent(value) {
        return `${Math.round((value || 0) * 100)}%`;
    }

    function formatScore(value) {
        if (value === undefined || value === null || Number.isNaN(Number(value))) {
            return "-";
        }
        return Number(value).toFixed(2);
    }

    function setBarcodeState(value, status) {
        barcodeValue.textContent = value || "No barcode scanned yet.";
        barcodeStatus.textContent = status || "Waiting for scanner input.";
    }

    function renderResult(result) {
        if (!result) {
            resultEmptyState.classList.remove("hidden");
            resultContent.classList.add("hidden");
            return;
        }

        const label = result.label || result.prediction;
        const confidence = result.confidence;
        const imageUrl = result.image_url;
        const extractedText = result.ocr_text;
        const detailedScores = result.detailed_scores || {};

        resultEmptyState.classList.add("hidden");
        resultContent.classList.remove("hidden");
        verdictCard.classList.remove("real", "fake");
        verdictCard.classList.add(String(label).toLowerCase());
        resultLabel.textContent = label;
        resultConfidence.textContent = formatPercent(confidence);
        resultImage.src = imageUrl;
        resultImage.classList.remove("hidden");
        ocrText.textContent = extractedText || "No readable text found.";
        resnetScore.textContent = formatScore(detailedScores.resnet_score);
        efficientnetScore.textContent = formatScore(detailedScores.efficientnet_score);
        vitScore.textContent = formatScore(detailedScores.vit_score);
        ocrScore.textContent = formatScore(detailedScores.ocr_score);
        ensembleScore.textContent = formatScore(detailedScores.ensemble_score);
        historySavedChip.classList.remove("hidden");
        resultContent.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    async function submitScan(payload, isFormData) {
        setProcessingState(true);
        setFeedback("Processing image through the prediction pipeline...", "loading");

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                body: isFormData ? payload : JSON.stringify(payload),
                headers: isFormData ? {} : { "Content-Type": "application/json" },
            });

            const data = await response.json();
            if (!response.ok || !data.ok) {
                throw new Error(data.error || "Scan failed.");
            }

            state.latestResult = data.result || data;
            renderResult(state.latestResult);
            setFeedback("Processing complete. Result displayed and saved to history.", "success");
        } finally {
            setProcessingState(false);
        }
    }

    async function sendBarcodeValue(decodedText) {
        const response = await fetch("/scan-barcode", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ barcode_value: decodedText }),
        });

        const data = await response.json();
        if (!response.ok || !data.ok) {
            throw new Error(data.error || "Barcode scan failed.");
        }

        setBarcodeState(
            data.barcode.barcode_value,
            `Captured and sent to Flask. Blockchain stage: ${data.barcode.verification_stage}.`
        );
    }

    uploadForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (isProcessing) {
            return;
        }

        try {
            await submitScan(new FormData(uploadForm), true);
            uploadForm.reset();
        } catch (error) {
            setFeedback(error.message, "error");
        }
    });

    startCameraButton.addEventListener("click", async () => {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            setFeedback("Camera access is not supported in this browser.", "error");
            return;
        }

        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "environment" },
                audio: false,
            });
            cameraFeed.srcObject = cameraStream;
            setFeedback("Live camera preview started.", "success");
        } catch (error) {
            if (error && (error.name === "NotAllowedError" || error.name === "SecurityError")) {
                setFeedback("Camera access was denied. Please allow camera permission in your browser and try again.", "error");
                return;
            }

            setFeedback("Unable to access camera. Check permissions and try again.", "error");
        }
    });

    captureButton.addEventListener("click", async () => {
        if (isProcessing) {
            return;
        }

        if (!cameraStream) {
            setFeedback("Start the camera before capturing.", "error");
            return;
        }

        if (!cameraFeed.videoWidth || !cameraFeed.videoHeight) {
            setFeedback("Camera is still initializing. Please wait.", "error");
            return;
        }

        cameraCanvas.width = cameraFeed.videoWidth;
        cameraCanvas.height = cameraFeed.videoHeight;

        const context = cameraCanvas.getContext("2d");
        context.drawImage(cameraFeed, 0, 0, cameraCanvas.width, cameraCanvas.height);

        try {
            await submitScan({ image_data: cameraCanvas.toDataURL("image/jpeg", 0.92) }, false);
        } catch (error) {
            setFeedback(error.message, "error");
        }
    });

    async function startBarcodeScanner() {
        if (barcodeActive) {
            setBarcodeState(barcodeValue.textContent, "Barcode scanner is already running.");
            return;
        }

        if (!window.Html5Qrcode) {
            setBarcodeState(barcodeValue.textContent, "Barcode library failed to load.");
            return;
        }

        barcodeScanner = new Html5Qrcode("barcodeReader");

        try {
            await barcodeScanner.start(
                { facingMode: "environment" },
                { fps: 10, qrbox: { width: 250, height: 120 } },
                async (decodedText) => {
                    if (!barcodeActive) {
                        return;
                    }

                    barcodeActive = false;
                    try {
                        await barcodeScanner.stop();
                    } catch (error) {
                    }

                    try {
                        await sendBarcodeValue(decodedText);
                    } catch (error) {
                        setBarcodeState(decodedText, error.message);
                    }
                },
                () => {}
            );

            barcodeActive = true;
            setBarcodeState(barcodeValue.textContent, "Scanner active. Point the camera at a barcode or QR code.");
        } catch (error) {
            barcodeActive = false;
            setBarcodeState(barcodeValue.textContent, "Unable to start barcode scanner.");
        }
    }

    async function stopBarcodeScanner() {
        if (!barcodeScanner || !barcodeActive) {
            setBarcodeState(barcodeValue.textContent, "Scanner is not currently running.");
            return;
        }

        try {
            barcodeActive = false;
            await barcodeScanner.stop();
            setBarcodeState(barcodeValue.textContent, "Barcode scanner stopped.");
        } catch (error) {
            setBarcodeState(barcodeValue.textContent, "Could not stop the barcode scanner cleanly.");
        }
    }

    startBarcodeButton.addEventListener("click", async () => {
        if (isProcessing) {
            return;
        }
        await startBarcodeScanner();
    });

    stopBarcodeButton.addEventListener("click", async () => {
        await stopBarcodeScanner();
    });

    renderResult(state.latestResult);
    setFeedback("Choose upload or camera capture to begin scanning.", "success");
    setBarcodeState("", "Waiting for scanner input.");
})();
