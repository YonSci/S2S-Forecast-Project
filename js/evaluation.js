async function loadEvaluation() {
    try {
        const response = await fetch("outputs/evaluation_report.json");
        if (!response.ok) throw new Error("Report not found");
        
        const data = await response.json();
        renderTable(data);
    } catch (err) {
        console.error("Failed to load evaluation report:", err);
        document.getElementById("eval-table-container").innerHTML = 
            `<p style="color: var(--text-secondary); text-align: center;">Unable to load evaluation report. Please ensure the pipeline has run.</p>`;
    }
}

function renderTable(data) {
    const container = document.getElementById("eval-table-container");
    const meta = data.meta;
    
    // Create Metadata Header
    let html = `
    <div style="margin-bottom: 2rem; text-align: center; color: var(--text-secondary);">
        <p><strong>Test Years:</strong> ${meta.test_years.join(", ")} &bull; 
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
        // Color skill based on value (positive/negative)
        // RMSE/MAE skill: Positive means better (1 - mod/clim) -> Good
        // ACC/HSS skill: Positive means better (mod - clim) -> Good
        // So always green if > 0, red if < 0.
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

loadEvaluation();
