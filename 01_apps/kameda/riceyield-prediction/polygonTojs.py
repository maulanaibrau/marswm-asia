import xml.etree.ElementTree as ET
import sys

args = sys.argv
#print(len(args))
"""
if(len(args)>=2):
	readNUM = int(args[1])
"""
#print(readNUM)

year = 2023

head_str="var areaRect" + str(year) + "= [\n\t{\n"

pre_str ="\t\t\"type\": \"Feature\",\n\t\t\"properties\": {\n"
pre_str =pre_str+"\t\t\t\"cellnum\": "
pre_my="\t\t\t\"predict\": "
pre2_str ="\t\t\t\"popupContent\": \"CN: \",\n\t\t\t\"style\": {\n"
pre2_str =pre2_str+"\t\t\t\tstroke: false,\n"
pre2_str =pre2_str+"\t\t\t\tweight: 0,\n\t\t\t\tcolor: \"#FFFFFF\",\n\t\t\t\topacity: 1,\n"
pre2_str =pre2_str+"\t\t\t\tfillColor: "
pre3_str = "\n\t\t\t\tfillOpacity: 1\n\t\t\t}\n\t\t},\n"
pre3_str =pre3_str+"\t\t\"geometry\": {\n\t\t\t\"type\": \"MultiPolygon\",\n\t\t\t\"coordinates\": [\n"
pre3_str =pre3_str+"\t\t\t\t[\n\t\t\t\t\t[\n"

post_str="\t\t\t\t\t]\n\t\t\t\t]\n\t\t\t]\n\t\t}\n\t},{\n"

foot_str="];\n"

name = 'Kamedagou_' + str(year) + '.kml'

tree = ET.parse(name)
#tree = ET.parse('地形適合Polygon_亀田郷04.kml')
root = tree.getroot()

cellData =[]
for line in root.iter('*'):
	if line.tag == '{http://www.opengis.net/kml/2.2}outerBoundaryIs':
		#print(line.find('{http://www.opengis.net/kml/2.2}LinearRing').find('{http://www.opengis.net/kml/2.2}coordinates').text)
		cellData.append(line.find('{http://www.opengis.net/kml/2.2}LinearRing').find('{http://www.opengis.net/kml/2.2}coordinates').text)
#	if len(cellData)==readNUM:
#		break

#print(len(cellData))
#print(cellData)

cellData2 =[]
for i, line in enumerate(cellData):
	ary_dat = line.split(" ")
#	print(i,": ",ary_dat)
	cellData2.append(ary_dat)

#print("NUM: ",len(cellData2))

id_Data=[]
for n in root.iter('*'):
	if n.tag == '{http://www.opengis.net/kml/2.2}SimpleData':
		if n.attrib.get('name')=='id':
			id_Data.append(n.text)

h_Data=[]
for m in root.iter('*'):
	if m.tag == '{http://www.opengis.net/kml/2.2}SimpleData':
		if m.attrib.get('name')=='Harvest':
			h_Data.append(float(m.text))

c_Data=[]
for p in range(len(h_Data)):
	if h_Data[p] < 500:
		c_Data.append(str("#FF0000"))
	elif 500 <= h_Data[p] < 550:
		c_Data.append(str("#FFA500"))
	elif 550 <= h_Data[p] < 600:
		c_Data.append(str("#F0E68C"))
	elif 600 <= h_Data[p] < 650:
		c_Data.append(str("#FFFF00"))
	elif 650 <= h_Data[p] < 700:
		c_Data.append(str("#90EE00"))
	elif 700 <= h_Data[p] < 750:
		c_Data.append(str("#00FF00"))
	else:
		c_Data.append(str("#006400"))


'''
print(len(cellData))
print(len(cellData2))
print(len(id_Data))
print(len(h_Data))
print(len(c_Data))
'''

print(head_str)
d_str=""
for i, line in enumerate(cellData2):
	d_str = d_str+pre_str+id_Data[i]+",\n"
	d_str = d_str+pre_my+str(h_Data[i])+",\n"
	d_str = d_str+pre2_str+"\""+c_Data[i]+"\""+","
	d_str = d_str+pre3_str
	for line2 in line:
		d_str=d_str+"\t\t\t\t\t\t["+line2+"],\n"
	d_str=d_str[0:-2]+"\n"
	d_str=d_str+post_str
d_str=d_str[0:-3]+"\n"
print(d_str)
print(foot_str)


