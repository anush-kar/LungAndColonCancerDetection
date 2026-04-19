(function () {
  const microSection = document.getElementById("microplasticsSection");
  if (!microSection) return;

  const correlationBtn = document.getElementById("runCorrelationBtn");

  const microError = document.getElementById("microError");
  const correlationContainer = document.getElementById("microCorrelation");

  const correlationMeta = document.getElementById("microCorrelationMeta");
  const correlationCanvas = document.getElementById("microCorrelationChart");

  const modeSelect = document.getElementById("microMode");
  const correlationMethodSelect = document.getElementById("microMethod");
  const stationLimitInput = document.getElementById("microStationLimit");
  const incidenceRegionSelect = document.getElementById("incidenceRegion");
  const incidenceGenderSelect = document.getElementById("incidenceGender");
  const incidenceMetricSelect = document.getElementById("incidenceMetric");

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

  function drawCorrelationChart(points, trendline, xLabel, yLabel) {
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
    ctx.fillText(xLabel || "X", width / 2 - 45, height - 10);
    ctx.save();
    ctx.translate(14, height / 2 + 30);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel || "Y", 0, 0);
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
      const isSelected = Boolean(p.is_selected);
      ctx.fillStyle = isSelected ? "#e85d04" : "#4f46e5";
      ctx.beginPath();
      ctx.arc(px, py, isSelected ? 6 : 4, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  async function loadIncidenceMetadata() {
    try {
      const payload = await requestJson("/api/incidence-waqi/metadata");
      const regions = Array.isArray(payload?.data?.regions) ? payload.data.regions : [];

      incidenceRegionSelect.innerHTML = "";
      regions.forEach((region) => {
        const option = document.createElement("option");
        option.value = region;
        option.textContent = region;
        incidenceRegionSelect.appendChild(option);
      });

      if (regions.length === 0) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No incidence regions found";
        incidenceRegionSelect.appendChild(option);
      }
    } catch (err) {
      setError(`Failed to load incidence metadata: ${err.message}`);
      incidenceRegionSelect.innerHTML = '<option value="">Metadata unavailable</option>';
    }
  }

  correlationBtn.addEventListener("click", async () => {
    setError("");
    correlationContainer.style.display = "none";

    const parsedLimit = Number.parseInt(stationLimitInput?.value || "4", 10);
    const station_limit = Number.isFinite(parsedLimit) ? Math.min(8, Math.max(1, parsedLimit)) : 4;

    try {
      const payload = await requestJson("/api/incidence-waqi/correlation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          region: incidenceRegionSelect?.value || "",
          gender: incidenceGenderSelect?.value || "combined",
          incidence_metric: incidenceMetricSelect?.value || "aar",
          waqi_mode: modeSelect.value,
          method: correlationMethodSelect.value,
          station_limit,
        }),
      });

      const c = payload.correlation;
      const strength = payload.interpretation?.strength || "N/A";
      const coverage = payload.coverage || {};
      const selected = payload.selected_region_point;
      const selectedText = selected
        ? `${selected.region} -> X=${selected.x.toFixed(2)}, Y=${selected.y.toFixed(2)}, Stations=${selected.station_count}`
        : `${c.selected_region || "N/A"} not matched in valid point set`;

      correlationMeta.innerHTML = `
        <strong>Selected Region:</strong> ${selectedText}<br>
        <strong>Method:</strong> ${c.method}<br>
        <strong>X Metric:</strong> ${c.x_metric}<br>
        <strong>Y Metric:</strong> ${c.y_metric}<br>
        <strong>Coverage:</strong> ${coverage.regions_used}/${coverage.regions_total} regions<br>
        <strong>Skipped (no station match):</strong> ${coverage.skipped_no_station_match ?? 0}<br>
        <strong>Skipped (missing data):</strong> ${coverage.skipped_missing_data ?? 0}<br>
        <strong>Sample Size:</strong> ${c.n}<br>
        <strong>Correlation (r):</strong> ${c.r}<br>
        <strong>P-Value:</strong> ${c.p_value ?? "N/A"}<br>
        <strong>Strength:</strong> ${strength}
      `;
      drawCorrelationChart(payload.points || [], c.trendline || null, c.x_metric, c.y_metric);
      correlationContainer.style.display = "block";
    } catch (err) {
      setError(err.message);
    }
  });

  loadIncidenceMetadata();
})();
