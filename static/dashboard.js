(function () {
    const state = window.__PHARMA_DASHBOARD__ || { latestResult: null, history: [], stats: {} };

    const uploadForm = document.getElementById("uploadForm");
    const feedbackEl = document.getElementById("scanFeedback");
    const resultEmptyState = document.getElementById("resultEmptyState");
    const resultContent = document.getElementById("resultContent");
    const resultLabel = document.getElementById("resultLabel");
    const resultConfidence = document.getElementById("resultConfidence");
    const resultImage = document.getElementById("resultImage");
    const ocrText = document.getElementById("ocrText");
    const historyTableBody = document.getElementById("historyTableBody");
    const verdictCard = document.getElementById("verdictCard");
    const cameraFeed = document.getElementById("cameraFeed");
    const cameraCanvas = document.getElementById("cameraCanvas");
    const startCameraButton = document.getElementById("startCameraButton");
    const captureButton = document.getElementById("captureButton");
    let cameraStream = null;

    function setFeedback(message, type) {
        feedbackEl.textContent = message;
        feedbackEl.className = "feedback";
        if (type) feedbackEl.classList.add(type);
    }

    function formatPercent(value) {
        return `${Math.round((value || 0) * 100)}%`;
    }

    function escapeHtml(value) {
        return String(value || "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#39;");
    }

    function capitalize(value) {
        return value ? value.charAt(0).toUpperCase() + value.slice(1) : "";
    }

    function renderStats(stats) {
        document.querySelectorAll("[data-stat]").forEach((node) => {
            const key = node.getAttribute("data-stat");
            if (key in stats) node.textContent = stats[key];
        });
    }

    function renderHistory(history) {
        if (!history || history.length === 0) {
            historyTableBody.innerHTML = '<tr><td colspan="6" class="history-empty">No scans recorded yet.</td></tr>';
            return;
        }

        historyTableBody.innerHTML = history.map((item) => `
            <tr>
                <td>#${item.id}</td>
                <td>
                    <div class="history-image-cell">
                        <img src="${item.image_url}" alt="${escapeHtml(item.image_name)}">
                        <span>${escapeHtml(item.image_name)}</span>
                    </div>
                </td>
                <td>${escapeHtml(capitalize(item.source))}</td>
                <td><span class="pill ${item.label.toLowerCase()}">${escapeHtml(item.label)}</span></td>
                <td>${formatPercent(item.confidence)}</td>
                <td>${escapeHtml(item.created_at)}</td>
            </tr>
        `).join("");
    }

    function renderResult(result) {
        if (!result) {
            resultEmptyState.classList.remove("hidden");
            resultContent.classList.add("hidden");
            return;
        }

        resultEmptyState.classList.add("hidden");
        resultContent.classList.remove("hidden");
        verdictCard.classList.remove("real", "fake");
        verdictCard.classList.add(result.label.toLowerCase());
        resultLabel.textContent = result.label;
        resultConfidence.textContent = formatPercent(result.confidence);
        resultImage.src = result.image_url;
        resultImage.classList.remove("hidden");
        ocrText.textContent = result.ocr_text || "No readable text found.";
    }

    async function submitScan(payload, isFormData) {
        setFeedback("Running medicine verification...", "loading");

        const response = await fetch("/api/scan", {
            method: "POST",
            body: isFormData ? payload : JSON.stringify(payload),
            headers: isFormData ? {} : { "Content-Type": "application/json" },
        });

        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || "Scan failed.");

        state.latestResult = data.result;
        state.history = data.history;
        state.stats = data.stats;
        renderResult(state.latestResult);
        renderHistory(state.history);
        renderStats(state.stats);
        setFeedback(`Verification complete: ${data.result.label}.`, "success");
    }

    uploadForm.addEventListener("submit", async (event) => {
        event.preventDefault();
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
            setFeedback("Camera started successfully.", "success");
        } catch (error) {
            setFeedback("Unable to access camera. Check permissions and try again.", "error");
        }
    });

    captureButton.addEventListener("click", async () => {
        if (!cameraStream) {
            setFeedback("Start the camera before capture.", "error");
            return;
        }

        if (!cameraFeed.videoWidth || !cameraFeed.videoHeight) {
            setFeedback("Camera is still initializing. Please wait.", "error");
            return;
        }

        cameraCanvas.width = cameraFeed.videoWidth;
        cameraCanvas.height = cameraFeed.videoHeight;
        cameraCanvas.getContext("2d").drawImage(cameraFeed, 0, 0);

        try {
            await submitScan({ image_data: cameraCanvas.toDataURL("image/jpeg", 0.92) }, false);
        } catch (error) {
            setFeedback(error.message, "error");
        }
    });

    document.querySelectorAll(".tab-button").forEach((button) => {
        button.addEventListener("click", () => {
            const mode = button.dataset.mode;
            document.querySelectorAll(".tab-button").forEach((node) => node.classList.toggle("active", node === button));
            document.querySelectorAll(".scan-pane").forEach((pane) => pane.classList.toggle("active", pane.dataset.pane === mode));
        });
    });

    document.querySelectorAll(".nav-link").forEach((button) => {
        button.addEventListener("click", () => {
            const target = button.dataset.section;
            document.querySelectorAll(".nav-link").forEach((node) => node.classList.toggle("active", node === button));
            document.querySelectorAll(".page-section").forEach((section) => section.classList.toggle("active", section.id === target));
        });
    });

    renderStats(state.stats || {});
    renderHistory(state.history || []);
    renderResult(state.latestResult);
    setFeedback("Dashboard ready. Upload or capture a medicine image to begin.", "success");
})();
