function openModal(src, caption) {
    var modal = document.getElementById("myModal");
    var modalImg = document.getElementById("modalImg");
    var modalCaption = document.getElementById("modalCaption");

    modal.style.display = "block";
    modalImg.src = src;
    modalCaption.innerHTML = caption;
}

function closeModal() {
    var modal = document.getElementById("myModal");
    modal.style.display = "none";
}

// Close the modal if the user clicks anywhere outside of it
window.onclick = function (event) {
    const modal = document.getElementById("myModal");
    if (event.target == modal) {
        closeModal();
    }   
}
// Close the modal if user hits 'Esc'
document.addEventListener('keydown', function(event) {
    if (event.key === "Escape") {
        closeModal();
    }
});
