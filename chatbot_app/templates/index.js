`use strict`

function formatDate(date) {
    const h = "0" + date.getHours();
    const m = "0" + date.getMinutes();

    return `${h.slice(-2)}:${m.slice(-2)}`;
}

var datetime = formatDate(new Date());
console.log(datetime);
document.getElementById("time").textContent = datetime; //it will print on html page
