const leadSelect = document.getElementById("lead-select");
const dateSelect = document.getElementById("date-select");
const linkNc = document.getElementById("link-nc");

const imgMap = {
  tercile: document.getElementById("img-tercile"),
  anomaly: document.getElementById("img-anomaly"),
  total: document.getElementById("img-total"),
  percent: document.getElementById("img-percent"),
};

let forecastManifest = null;

async function init() {
  try {
    const response = await fetch("outputs/forecast_index.json");
    forecastManifest = await response.json();

    // Initial populate
    updateDateOptions();
    
    // Initial load
    updateForecast();

    if (leadSelect) {
        leadSelect.addEventListener("change", () => {
          updateDateOptions();
          updateForecast();
        });
    }
    
    if (dateSelect) {
        dateSelect.addEventListener("change", () => {
          updateForecast();
        });
    }
  } catch (err) {
    console.error("Failed to load forecast manifest:", err);
  }
}

// Format date nicely (e.g. "Dec 23")
function formatSimpleDate(date) {
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function updateDateOptions() {
  if (!leadSelect || !dateSelect) return;

  const lead = leadSelect.value;
  const dates = forecastManifest.leads[lead] || [];
  
  // Clear existing options
  dateSelect.innerHTML = "";
  
  if (dates.length === 0) {
    const option = document.createElement("option");
    option.text = "No forecasts";
    dateSelect.add(option);
    dateSelect.disabled = true;
    return;
  }
  
  dateSelect.disabled = false;
  dates.forEach(dateStr => {
      // Calculate week range
      const startDate = new Date(dateStr);
      const endDate = new Date(startDate);
      endDate.setDate(startDate.getDate() + 6);
      
      const label = `${formatSimpleDate(startDate)} - ${formatSimpleDate(endDate)}`;
      
      const option = document.createElement("option");
      option.value = dateStr;
      option.text = label; 
      dateSelect.add(option);
  });
}

function updateForecast() {
  if (!leadSelect || !dateSelect) return;

  const lead = leadSelect.value;
  const selectedDate = dateSelect.value;
  
  if (!selectedDate || selectedDate === "No forecasts") {
    console.warn(`No valid date selected for lead ${lead}`);
    return;
  }

  const ts = new Date().getTime(); // Cache-buster

  // Update all images
  for (const [key, img] of Object.entries(imgMap)) {
    if (img) {
        img.src = `outputs/forecast_${selectedDate}_${lead}_${key}.png?v=${ts}`;
    }
  }
  // Update NetCDF download link
  if (linkNc) {
    linkNc.href = `outputs/forecast_${selectedDate}_${lead}.nc?v=${ts}`;
  }
}

init();
