const btn = document.getElementById('addPatient');
const modal = document.getElementById('patientModal');
const modalContent = document.getElementById('patientModalContent');

function addShowHideModal() {
  btn.onclick = function(event) {
    modal.style.display = "block";
    console.log("Modal shown");
  }

  window.onclick = function(event) {
    // Close the modal if the backdrop is clicked
    // Excluding the button prevents an open and immediate close when clicking button
    if (event.target == modal && event.target != btn && modal.hidden == false) {
      modal.style.display = "none";
      console.log("Modal hidden");
    }
  }
}
