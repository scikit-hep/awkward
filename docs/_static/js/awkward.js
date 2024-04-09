document.addEventListener("DOMContentLoaded", function() {
    document.querySelectorAll('a[href="_static/try-it.html"]').forEach(a => {
        a.target = "_blank";
    });
});
