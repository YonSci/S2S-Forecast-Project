const themeToggle = document.getElementById("theme-toggle");
const html = document.documentElement;

function setTheme(theme) {
  html.setAttribute("data-theme", theme);
  localStorage.setItem("theme", theme);
  updateThemeIcon(theme);
}

function updateThemeIcon(theme) {
  const moon = document.getElementById("moon-path");
  if (!moon) return; 

  const sunElements = [
    "sun-circle", "sun-l1", "sun-l2", "sun-l3", 
    "sun-l4", "sun-l5", "sun-l6", "sun-l7", "sun-l8"
  ];

  if (theme === "light") {
    moon.style.display = "none";
    sunElements.forEach(id => {
        const el = document.getElementById(id);
        if(el) el.style.display = "block";
    });
  } else {
    moon.style.display = "block";
    sunElements.forEach(id => {
        const el = document.getElementById(id);
        if(el) el.style.display = "none";
    });
  }
}

if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const currentTheme = html.getAttribute("data-theme");
      setTheme(currentTheme === "dark" ? "light" : "dark");
    });
}

// Initialize theme
const savedTheme = localStorage.getItem("theme") || "dark";
setTheme(savedTheme);
