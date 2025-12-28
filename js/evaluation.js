const yearSelect = document.getElementById("eval-year-select");

async function loadEvaluation(year) {
    const yearSelect = document.getElementById("eval-year-select");
    // defaults to currently selected or 2020
    year = year || (yearSelect ? yearSelect.value : "2020") || "2020";
    
    // Update Images
    updateImages(year);

    // Update Interactive Plots
    loadInteractivePlots(year);
    
    // Update Report Table
    try {
        const response = await fetch(`outputs/evaluation_report_${year}.json`);
        if (!response.ok) throw new Error(`Report for ${year} not found`);
        
        const data = await response.json();
        renderTable(data);
        
        // Update Title
        document.getElementById("spatial-title").innerText = `Spatial Performance Analysis (${year})`;
        
    } catch (err) {
        console.error("Failed to load evaluation report:", err);
        document.getElementById("eval-table-container").innerHTML = 
            `<p style="color: var(--text-secondary); text-align: center;">Unable to load evaluation report for ${year}. Please ensure the pipeline has run.</p>`;
    }
}

function updateImages(year) {
    const images = [
        "img-spatial-bias",
        "img-spatial-rmse",
        "img-spatial-acc",
        "img-spatial-hitrate",
        "img-confusion-matrix",
        "img-extreme-confusion-matrix",
        "img-scatter-plot"
    ];
    
    images.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            // Update src, e.g., outputs/spatial_bias_2020.png
            // We assume the ID format is img-{name} and file is outputs/{name}_{year}.png
            // Actually simpler: just replace the year part of the existing src or rebuild it.
            // Let's rebuild strictly to avoid errors.
            const name = id.replace("img-", "").replace(/-/g, "_"); // "spatial-bias" -> "spatial_bias"
            el.src = `outputs/${name}_${year}.png`;
            
            // Add error handler fallback
            el.onerror = () => {
                // el.src = "assets/placeholder.png"; // Optional fallback
                console.warn(`Missing image: ${el.src}`);
            };
        }
    });
}

function renderTable(data) {
    const container = document.getElementById("eval-table-container");
    const meta = data.meta;
    
    // Create Metadata Header
    let html = `
    <div style="margin-bottom: 2rem; text-align: center; color: var(--text-secondary);">
        <p><strong>Test Years:</strong> ${meta.year} &bull; 
           <strong>Samples:</strong> ${meta.samples} &bull; 
           <strong>Lead Time:</strong> ${meta.lead_weeks} Week(s)</p>
    </div>
    <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; color: var(--text-primary);">
            <thead>
                <tr style="border-bottom: 2px solid var(--card-border);">
                    <th style="padding: 1rem; text-align: left;">Metric</th>
                    <th style="padding: 1rem; text-align: right;">Model</th>
                    <th style="padding: 1rem; text-align: right;">Persistence</th>
                    <th style="padding: 1rem; text-align: right;">Climatology</th>
                    <th style="padding: 1rem; text-align: right;">Skill (vs Clim)</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    data.metrics.forEach(row => {
        const skillColor = row.skill > 0 ? "#3fb950" : "#f85149";
        
        html += `
            <tr style="border-bottom: 1px solid var(--card-border);">
                <td style="padding: 1rem; font-weight: 600;">${row.name}</td>
                <td style="padding: 1rem; text-align: right; font-family: monospace;">${row.model.toFixed(4)}</td>
                <td style="padding: 1rem; text-align: right; font-family: monospace; color: var(--text-secondary);">${row.persistence.toFixed(4)}</td>
                <td style="padding: 1rem; text-align: right; font-family: monospace; color: var(--text-secondary);">${row.climatology.toFixed(4)}</td>
                <td style="padding: 1rem; text-align: right; font-family: monospace; font-weight: 700; color: ${skillColor};">
                    ${row.skill > 0 ? "+" : ""}${row.skill.toFixed(4)}
                </td>
            </tr>
        `;
    });
    
    html += `
            </tbody>
        </table>
    </div>
    `;
    
    container.innerHTML = html;
}

// Event Listeners
if (yearSelect) {
    yearSelect.addEventListener("change", (e) => {
        loadEvaluation(e.target.value);
    });
}

const downloadBtn = document.getElementById("download-report-btn");
if (downloadBtn) {
    downloadBtn.addEventListener("click", () => {
        const year = yearSelect.value;
        const filename = `evaluation_report_${year}.json`;
        const filepath = `outputs/${filename}`;
        
        // Create temporary link
        const link = document.createElement('a');
        link.href = filepath;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
}

// Initial Load
async function init() {
    try {
        const response = await fetch('outputs/manifest.json');
        if (response.ok) {
            const manifest = await response.json();
            
            // Clear existing options
            yearSelect.innerHTML = '';
            
            // Populate
            manifest.available_years.forEach(year => {
                const opt = document.createElement('option');
                opt.value = year;
                opt.textContent = year;
                yearSelect.appendChild(opt);
            });
            
            // Select latest by default or stick to current
            const latest = manifest.latest_year;
            if (latest) {
                yearSelect.value = latest;
                loadEvaluation(latest);
            } else {
                // Fallback
                loadEvaluation("2020");
            }
        } else {
            console.warn("Manifest not found, using static defaults.");
        }
    } catch (err) {
        console.error("Error loading manifest:", err);
    }
    
    // Fallback if manifest fails or was empty
    if (yearSelect.options.length === 0) {
        // Manually add at least 2020
        const opt = document.createElement('option');
        opt.value = "2020";
        opt.textContent = "2020";
        yearSelect.appendChild(opt);
        loadEvaluation("2020");
    }
}

init();

// Map Layer Toggles
// Map Layer Toggles
const toggleCountry = document.getElementById('toggle-country');
const toggleRegions = document.getElementById('toggle-regions');
const toggleZones = document.getElementById('toggle-zones');

// Force refresh overlays to clear cache
const overlays = document.querySelectorAll('.overlay-layer');
overlays.forEach(img => {
    const currentSrc = img.getAttribute('src').split('?')[0];
    img.src = `${currentSrc}?v=${new Date().getTime()}`;
});

function updateOverlayVisibility(className, isChecked) {
    const overlays = document.querySelectorAll(className);
    overlays.forEach(img => img.style.opacity = isChecked ? '1' : '0');
}

if (toggleCountry) {
    toggleCountry.addEventListener('change', function() {
        updateOverlayVisibility('.country-overlay', this.checked);
    });
    // Set initial state
    updateOverlayVisibility('.country-overlay', toggleCountry.checked);
}

if (toggleRegions) {
    toggleRegions.addEventListener('change', function() {
        updateOverlayVisibility('.region-overlay', this.checked);
    });
    // Set initial state
    updateOverlayVisibility('.region-overlay', toggleRegions.checked);
}

if (toggleZones) {
    toggleZones.addEventListener('change', function() {
        updateOverlayVisibility('.zone-overlay', this.checked);
    });
    // Set initial state
    updateOverlayVisibility('.zone-overlay', toggleZones.checked);
}

// --- Plotly Integration ---
async function loadInteractivePlots(year) {
    // 1. Temporal
    try {
        const response = await fetch(`outputs/temporal_metrics_${year}.json`);
        if (response.ok) {
            const data = await response.json();
            renderTemporalErrorPlot(data, year);
            renderTemporalSkillPlot(data, year);
        }
    } catch (err) { console.warn("Skipping temporal:", err); }

    // 2. Matrices
    loadMatrixPlots(year);
    
    // 3. Scatter
    loadScatterMetric(year);
}

async function loadMatrixPlots(year) {
    try {
        // Confusion Matrix
        const resp1 = await fetch(`outputs/confusion_matrix_${year}.json`);
        if (resp1.ok) renderConfusionMatrix(await resp1.json(), year);
        
        // Extreme Matrix
        const resp2 = await fetch(`outputs/extreme_confusion_matrix_${year}.json`);
        if (resp2.ok) renderExtremeMatrix(await resp2.json(), year);
    } catch (err) { console.warn("Skipping matrices:", err); }
}

async function loadScatterMetric(year) {
    try {
        const response = await fetch(`outputs/scatter_metrics_${year}.json`);
        if (response.ok) renderScatterPlot(await response.json(), year);
    } catch (err) { console.warn("Skipping scatter:", err); }
}

function renderTemporalErrorPlot(data, year) {
    const dates = data.map(d => d.Date);
    const rmse = data.map(d => d.RMSE_Model);
    const mae = data.map(d => d.MAE_Model);
    
    const trace1 = {
        x: dates, y: rmse,
        mode: 'lines+markers',
        name: 'RMSE (Model)',
        line: { color: '#ef4444', width: 2 },
        marker: { size: 6 }
    };
    
    const trace2 = {
        x: dates, y: mae,
        mode: 'lines+markers',
        name: 'MAE (Model)',
        line: { color: '#f97316', width: 2, dash: 'dash' },
        marker: { size: 6 }
    };
    
    const layout = getPlotlyLayout(`Error Metric Evolution (RMSE & MAE) - ${year}`, "Error (mm/day)");
    layout.xaxis.tickangle = -90;
    layout.xaxis.dtick = "M1";
    layout.xaxis.tickformat = "%d-%m-%Y";
    layout.hovermode = 'x unified';
    layout.hoverlabel = { font: { family: "Arial, sans-serif", size: 13 } };
    
    Plotly.newPlot('plot-temporal-error', [trace1, trace2], layout, {responsive: true, displayModeBar: false});
}

function renderTemporalSkillPlot(data, year) {
    const dates = data.map(d => d.Date);
    // Assuming CSV has ACC_Model and HitRate_Model
    const acc = data.map(d => d.ACC_Model);
    const hitrate = data.map(d => d.HitRate_Model);
    
    const trace1 = {
        x: dates, y: acc,
        mode: 'lines+markers',
        name: 'ACC (Model)',
        line: { color: '#3b82f6', width: 2 },
        marker: { size: 6 }
    };
    
    const trace2 = {
        x: dates, y: hitrate,
        mode: 'lines+markers',
        name: 'Hit Rate (Model)',
        line: { color: '#22c55e', width: 2 },
        marker: { size: 6 }
    };
    
    const layout = getPlotlyLayout(`Skill Score Evolution (ACC & Hit Rate) - ${year}`, "Score");
    layout.yaxis.range = [-1.1, 1.1]; // ACC is -1 to 1, HitRate 0 to 1
    layout.xaxis.tickangle = -90;
    layout.xaxis.dtick = "M1";
    layout.xaxis.tickformat = "%d-%m-%Y";
    layout.hovermode = 'x unified';
    layout.hoverlabel = { font: { family: "Arial, sans-serif", size: 13 } };
    
    Plotly.newPlot('plot-temporal-skill', [trace1, trace2], layout, {responsive: true, displayModeBar: false});
}

function renderConfusionMatrix(data, year) {
    const layout = getPlotlyLayout(`Tercile Confusion Matrix - ${year}`, "Observed");
    layout.xaxis.title = { text: "Predicted" };
    layout.yaxis.autorange = "reversed"; 
    
    const trace = {
        z: data.matrix,
        x: data.labels,
        y: data.labels,
        type: 'heatmap',
        colorscale: 'Blues',
        text: data.matrix.map(row => row.map(v => (v*100).toFixed(1) + '%')),
        texttemplate: "%{text}",
        showscale: true,
        hovertemplate: "Predicted: %{x}<br>Observed: %{y}<br>Rate: %{text}<extra></extra>",
        hoverlabel: { font: { family: "Arial, sans-serif" } }
    };
    
    Plotly.newPlot('plot-confusion-matrix', [trace], layout, {responsive: true});
}

function renderExtremeMatrix(data, year) {
    const layout = getPlotlyLayout(`Extreme Event (Top 10%) Matrix - ${year}`, "Observed");
    layout.xaxis.title = { text: "Predicted" };
    layout.yaxis.autorange = "reversed";
    
    const trace = {
        z: data.matrix,
        x: data.labels,
        y: data.labels,
        type: 'heatmap',
        colorscale: 'Reds',
        text: data.matrix.map(row => row.map(v => (v*100).toFixed(1) + '%')),
        texttemplate: "%{text}",
        showscale: true,
        hovertemplate: "Predicted: %{x}<br>Observed: %{y}<br>Rate: %{text}<extra></extra>",
        hoverlabel: { font: { family: "Arial, sans-serif" } }
    };
    
    Plotly.newPlot('plot-extreme-confusion-matrix', [trace], layout, {responsive: true});
}

function renderScatterPlot(data, year) {
    const isDark = document.body.classList.contains('dark-mode') || document.documentElement.getAttribute('data-theme') === 'dark';
    const textColor = isDark ? '#e2e8f0' : '#1e293b';
    const gridColor = isDark ? '#334155' : '#cbd5e1';

    // Coerce data to numbers to handle potential type issues
    const xData = (data.observed || []).map(v => Number(v));
    const yData = (data.forecast || []).map(v => Number(v));
    const validCount = xData.filter(v => !isNaN(v)).length;

    const layout = {
        title: { text: `Forecast vs Obs (Grid Points) - ${year} (R=${data.correlation.toFixed(3)}, N=${validCount})`, font: { color: textColor, size: 16 } },
        font: { family: "Outfit, sans-serif" },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            title: { text: "Observed Mean (mm/day)", font: { family: "Outfit, sans-serif" } },
            color: textColor,
            gridcolor: gridColor,
            showgrid: true,
            type: 'linear'
        },
        yaxis: {
            title: { text: "Forecast Mean (mm/day)", font: { family: "Outfit, sans-serif" } },
            color: textColor,
            gridcolor: gridColor,
            showgrid: true,
            type: 'linear'
        },
        margin: { t: 40, r: 20, l: 60, b: 60 },
        hoverlabel: {
            bgcolor: isDark ? '#1e293b' : '#ffffff',
            bordercolor: gridColor,
            font: { color: textColor, size: 13, family: "Arial, sans-serif" }
        }
    };
    
    // Add diagonal 1:1 line matching data range
    const allVals = [...xData, ...yData].filter(v => Number.isFinite(v));
    const minVal = allVals.length ? Math.min(...allVals) : 0;
    const maxVal = allVals.length ? Math.max(...allVals) : 1;

    layout.shapes = [{
        type: 'line',
        x0: minVal, y0: minVal,
        x1: maxVal, y1: maxVal,
        line: { color: 'grey', dash: 'dash', width: 1.5 }
    }];
    
    const trace = {
        x: xData,
        y: yData,
        mode: 'markers',
        type: 'scatter',
        marker: { size: 8, color: '#3b82f6', opacity: 0.6 },
        name: 'Samples',
        hovertemplate: "Obs: %{x:.2f}<br>Fcst: %{y:.2f}<extra></extra>"
    };
    
    Plotly.newPlot('plot-scatter-metrics', [trace], layout, {responsive: true});
}

function getPlotlyLayout(title, yTitle) {
    const isDark = document.body.classList.contains('dark-mode') || document.documentElement.getAttribute('data-theme') === 'dark';
    const textColor = isDark ? '#e2e8f0' : '#1e293b';
    const gridColor = isDark ? '#334155' : '#cbd5e1';
    
    // Check if theme.js sets a class on body or logical attribute
    // Assuming standard dark mode handling, but can fallback to a neutral premium dark
    
    return {
        title: { text: title, font: { color: textColor, size: 16 } },
        font: { family: "Outfit, sans-serif" },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        hoverlabel: {
            bgcolor: isDark ? '#1e293b' : '#ffffff',
            bordercolor: gridColor,
            font: { color: textColor, size: 13, family: "Arial, sans-serif" }
        },
        margin: { b: 100, l: 60, r: 30, t: 50 },
        xaxis: {
            automargin: true,
            color: textColor,
            gridcolor: gridColor,
            showgrid: true,
            visible: true,
            title: { text: "", font: { family: "Outfit, sans-serif" } }
        },
        yaxis: {
            title: { text: yTitle },
            color: textColor,
            gridcolor: gridColor,
            showgrid: true,
            zeroline: false
        },
        legend: {
            font: { color: textColor },
            orientation: 'h',
            y: 1.1,
            x: 1,
            xanchor: 'right'
        },
        margin: { t: 40, r: 20, l: 60, b: 60 }
    };
}
