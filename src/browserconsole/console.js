
hijos = document.getElementsByClassName('mCSB_container')[0].childNodes[1]

function recurse(nodes, arreg){
	for (var i =0 ; i<nodes.length;i++){
		new_arra = arreg.slice(0);
		var childre = nodes[i].childNodes;
			if(childre.length<=2){
			new_arra.push(childre[1].text);
			console.log(childre[1].getAttribute('href')+":"+new_arra);
			//new_arra.reverse();
			///new_arra.push(childre[1].getAttribute('href'));
			//console.log("{"+childre[1].getAttribute('href')+ ':'+ new_arra+"}");
			}
			else{
				new_arra.push(childre[1].text);
			console.log(childre[1].getAttribute('href')+":"+new_arra);
			//console.log(new_arra);
			recurse(childre[2].childNodes,new_arra);
			}
		}
	}

for (var i = 0 ; i<hijos.length;i++)
{
	var nombre = hijos[i].childNodes[1].getAttribute('href');
	var nombretexto = hijos[i].childNodes[1].text;
	var hijo_sub = hijos[i].childNodes[2].childNodes;
	console.log(nombre+":"+nombretexto);
	var arreglo =[];
	arreglo.push(nombretexto);
	recurse(hijo_sub,arreglo);
}