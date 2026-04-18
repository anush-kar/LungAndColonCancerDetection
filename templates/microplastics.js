(function () {
  const microSection = document.getElementById("microplasticsSection");
  if (!microSection) return;

  const exposureBtn = document.getElementById("fetchExposureBtn");
  const correlationBtn = document.getElementById("runCorrelationBtn");
  const riskBtn = document.getElementById("runEnhancedRiskBtn");

  const microError = document.getElementById("microError");
  const exposureContainer = document.getElementById("microExposure");
  const correlationContainer = document.getElementById("microCorrelation");
  const riskContainer = document.getElementById("microRisk");

  const exposureGrid = document.getElementById("microExposureGrid");
  const correlationMeta = document.getElementById("microCorrelationMeta");
  const riskMeta = document.getElementById("microRiskMeta");
  const correlationCanvas = document.getElementById("microCorrelationChart");

  const cancerTypeSelect = document.getElementById("microCancerType");
  const correlationMethodSelect = document.getElementById("microMethod");

  async function requestJson(url, options = {}) {
    const response = await fetch(url, options);
    const payload = await response.json();
    if (!response.ok || payload.success === false) {
      throw new Error(payload.error || "Request failed");
    }
    return payload;
  }

  function setError(message) {
    microError.textContent = message || "";
    microError.style.display = message ? "block" : "none";
  }

  function selectedRegion() {
    const stationSelect = document.getElementById("stationSelect");
    const station = stationSelect?.value?.trim();
    if (station) {
      const parts = station.split(",");
      // Typical format: "Station, City, India" => use city for region matching.
      if (parts.length >= 2) {
        return parts[1].trim();
      }
      return station;
    }

    const legacyCitySelect = document.getElementById("state");
    return legacyCitySelect?.value?.trim() || "";
  }

  function addExposureItem(label, value) {
    const item = document.createElement("div");
    item.className = "aqi-chip";
    item.innerHTML = `
      <div class="aqi-chip-label">${label}</div>
      <div class="aqi-chip-value">${value ?? "N/A"}</div>
    `;
    exposureGrid.appendChild(item);
  }

  function drawCorrelationChart(points, trendline) {
    if (!correlationCanvas) return;

    const ctx = correlationCanvas.getContext("2d");
    const width = correlationCanvas.width;
    const height = correlationCanvas.height;
    const padding = { left: 48, right: 16, top: 16, bottom: 36 };

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    if (!Array.isArray(points) || points.length === 0) {
      ctx.fillStyle = "#666";
      ctx.font = "14px Segoe UI";
      ctx.fillText("No data points to plot", 20, 30);
      return;
    }

    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    let minX = Math.min(...xs);
    let maxX = Math.max(...xs);
    let minY = Math.min(...ys);
    let maxY = Math.max(...ys);

    if (minX === maxX) {
      minX -= 1;
      maxX += 1;
    }
    if (minY === maxY) {
      minY -= 1;
      maxY += 1;
    }

    const xPad = (maxX - minX) * 0.08;
    const yPad = (maxY - minY) * 0.12;
    minX -= xPad;
    maxX += xPad;
    minY -= yPad;
    maxY += yPad;

    const plotW = width - padding.left - padding.right;
    const plotH = height - padding.top - padding.bottom;

    const xToPx = (x) => padding.left + ((x - minX) / (maxX - minX)) * plotW;
    const yToPx = (y) => height - padding.bottom - ((y - minY) / (maxY - minY)) * plotH;

    ctx.strokeStyle = "#d0d0d0";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.stroke();

    ctx.fillStyle = "#666";
    ctx.font = "12px Segoe UI";
    ctx.fillText("Microplastics in Air", width / 2 - 45, height - 10);
    ctx.save();
    ctx.translate(14, height / 2 + 30);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Cancer Incidence", 0, 0);
    ctx.restore();

    if (trendline && trendline.slope != null && trendline.intercept != null) {
      const y1 = trendline.slope * minX + trendline.intercept;
      const y2 = trendline.slope * maxX + trendline.intercept;
      ctx.strokeStyle = "#ff6b6b";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(xToPx(minX), yToPx(y1));
      ctx.lineTo(xToPx(maxX), yToPx(y2));
      ctx.stroke();
    }

    points.forEach((p) => {
      const px = xToPx(p.x);
      const py = yToPx(p.y);
      ctx.fillStyle = "#4f46e5";
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  exposureBtn.addEventListener("click", async () => {
    const region = selectedRegion();
    setError("");
    exposureContainer.style.display = "none";
    if (!region) {
      setError("Please select a city/locality first.");
      return;
    }

    try {
      const payload = await requestJson(`/api/microplastics/exposure?region=${encodeURIComponent(region)}`);
      const data = payload.data;
      exposureGrid.innerHTML = "";
      addExposureItem("Region", data.region);
      addExposureItem("Year", data.year);
      addExposureItem("Microplastics (air)", data.microplastics_air);
      addExposureItem("PM2.5", data.pm25);
      addExposureItem("PM10", data.pm10);
      addExposureItem("Lung ACA incidence", data.lung_aca_incidence);
      addExposureItem("Lung SCC incidence", data.lung_scc_incidence);
      exposureContainer.style.display = "block";
    } catch (err) {
      setError(err.message);
    }
  });

  correlationBtn.addEventListener("click", async () => {
    setError("");
    correlationContainer.style.display = "none";

    try {
      const payload = await requestJson("/api/microplastics/correlation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cancer_type: cancerTypeSelect.value,
          method: correlationMethodSelect.value,
        }),
      });

      const c = payload.correlation;
      const strength = payload.interpretation?.strength || "N/A";
      correlationMeta.innerHTML = `
        <strong>Method:</strong> ${c.method}<br>
        <strong>Sample Size:</strong> ${c.n}<br>
        <strong>Correlation (r):</strong> ${c.r}<br>
        <strong>P-Value:</strong> ${c.p_value ?? "N/A"}<br>
        <strong>Strength:</strong> ${strength}
      `;
      drawCorrelationChart(payload.points || [], c.trendline || null);
      correlationContainer.style.display = "block";
    } catch (err) {
      setError(err.message);
    }
  });

  riskBtn.addEventListener("click", async () => {
    const region = selectedRegion();
    setError("");
    riskContainer.style.display = "none";

    if (!region) {
      setError("Please select a city/locality first.");
      return;
    }

    const predictedScore = window.latestImagePrediction?.score;
    if (predictedScore == null) {
      setError("Run image analysis first, then generate enhanced risk.");
      return;
    }

    try {
      const payload = await requestJson("/api/risk/enhanced", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          region,
          image_score: predictedScore,
          cancer_type: cancerTypeSelect.value,
        }),
      });

      const r = payload.risk;
      const factors = (r.explanation || []).map((f) => `<li>${f}</li>`).join("");
      riskMeta.innerHTML = `
        <strong>Image Risk:</strong> ${(r.base_image_risk * 100).toFixed(1)}%<br>
        <strong>Environmental Index:</strong> ${(r.environmental_index * 100).toFixed(1)}%<br>
        <strong>Combined Risk:</strong> ${(r.combined_risk * 100).toFixed(1)}%<br>
        <strong>Risk Tier:</strong> ${r.risk_tier}<br>
        <strong>Uncertainty:</strong> ±${(r.uncertainty * 100).toFixed(1)}%
        <ul style="margin-top:8px; padding-left:18px;">${factors}</ul>
      `;
      riskContainer.style.display = "block";
    } catch (err) {
      setError(err.message);
    }
  });
})();
