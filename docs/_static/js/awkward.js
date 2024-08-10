document.addEventListener("DOMContentLoaded", function() {
    document.querySelectorAll('a[href="_static/try-it.html"]').forEach(a => {
        a.target = "_blank";
    });
    document.querySelectorAll('a[href="https://awkward-array.org/doc/main/_static/try-it.html"]').forEach(a => {
        a.target = "_blank";
    });
});
