alert('script running')
document.getElementById("geturlbtn").addEventListener("click", ()=>{
    // alert('clicked')
    var tablink;
    alert('hi')
chrome.tabs.getSelected(null,function(tab) {
    alert('tablink')
    tablink = tab.url;
});
console.log(tablink);

})

