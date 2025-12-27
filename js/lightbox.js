document.addEventListener('DOMContentLoaded', () => {
    // 1. Create Lightbox Elements
    const lightbox = document.createElement('div');
    lightbox.id = 'lightbox';
    lightbox.className = 'lightbox-overlay';
    
    const closeBtn = document.createElement('button');
    closeBtn.innerHTML = '&times;';
    closeBtn.className = 'lightbox-close';
    closeBtn.title = "Close (Esc)";
    
    const imgContainer = document.createElement('div');
    imgContainer.className = 'lightbox-content';
    
    const lightboxImg = document.createElement('img');
    lightboxImg.className = 'lightbox-image';
    
    imgContainer.appendChild(lightboxImg);
    lightbox.appendChild(closeBtn);
    lightbox.appendChild(imgContainer);
    document.body.appendChild(lightbox);

    // 2. Function to Open Lightbox
    function openLightbox(src, alt) {
        lightboxImg.src = src;
        lightboxImg.alt = alt;
        lightbox.classList.add('active');
        document.body.style.overflow = 'hidden'; // Disable scroll
    }

    // 3. Function to Close Lightbox
    function closeLightbox() {
        lightbox.classList.remove('active');
        document.body.style.overflow = ''; // Re-enable scroll
    }

    // 4. Attach Event Listeners to all "Zoomable" Images
    // We target all images inside .tech-card, .forecast-card, or explicitly marked .zoomable
    const images = document.querySelectorAll('.tech-card img, .forecast-card img, img.zoomable');
    
    images.forEach(img => {
        if (img.classList.contains('overlay-layer')) return;
        
        img.classList.add('zoomable');
        img.addEventListener('click', (e) => {
            e.preventDefault();
            openLightbox(img.src, img.alt);
        });
    });

    // 5. Close Events
    closeBtn.addEventListener('click', closeLightbox);
    
    lightbox.addEventListener('click', (e) => {
        if (e.target === lightbox || e.target === imgContainer) {
            closeLightbox();
        }
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && lightbox.classList.contains('active')) {
            closeLightbox();
        }
    });
});
