const yearSelect = document.getElementById("eval-year-select");

async function loadEvaluation(year) {
    const yearSelect = document.getElementById("eval-year-select");
    // defaults to currently selected or 2020
    year = year || (yearSelect ? yearSelect.value : "2020") || "2020";
    
    // Update Images
    updateImages(year);
    
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
        "img-temporal-skill",
        "img-temporal-error",
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
