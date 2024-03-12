const mymap = L.map('mapid').setView([37.866,139.10],13);

const tiles = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{
        maxZoom:18.5,
        attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
    }).addTo(mymap);

function onEachFeature(feature, layer) {
/*	let popupContent = `<p>GeoJSON ${feature.geometry.type}</p>`;
	if (feature.properties && feature.properties.popupContent) {
		popupContent += feature.properties.popupContent;
	}
*/
	let popupContent = "Field ID: ";
	if (feature.properties && feature.properties.cellnum) {
		popupContent +=   feature.properties.cellnum + "<br>";
		popupContent += "Predicted Yield: " +  feature.properties.predict + "(kg/10a)" + "<br>";
	}
	if (feature.properties && feature.properties.style.fillColor) {
		popupContent += "Color: " + feature.properties.style.fillColor + "<br>";
	}
	
	layer.bindPopup(popupContent);
}

var Layer2023 = L.geoJSON(areaRect2023, {
	style(feature) {
		return feature.properties && feature.properties.style;
	},

	onEachFeature: onEachFeature
	
}).addTo(mymap);

var baseMaps = {
    "OpenStreetMap": tiles
};

var overlayMaps = {
	"2023": Layer2023
};

var layerControl = L.control.layers(baseMaps, overlayMaps).addTo(mymap);

var popup = L.popup();

function onMapClick(e) {
	popup
		.setLatLng(e.latlng)
		.setContent(`You clicked the map at ${e.latlng.toString()}`)
		.openOn(mymap);
}

mymap.on('click', onMapClick);

