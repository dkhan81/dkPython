colors = ["red","blue","green"]
print(colors[0])
print(colors[1])
print(colors[2])
print(len(colors))

color = ["red","blue","green"]
color2 = ["orange","black","white"]
print(color + color2)
print(len(color))
print(color[0])
color[0] = "yellow"
print(color[0])
print(color*2)
print("blue" in  color2)

color.append("white")
print(color)
color.extend(["black","purple"])
print(color)

color.insert(0,"orange")
print(color)
color.remove("white")
del color[0]
print(color)
