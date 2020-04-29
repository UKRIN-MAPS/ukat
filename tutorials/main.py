from B0Map import B0Map
from T2Star import T2Star

image = # 3D/4D numpy array
echoes = [1, 4, 7, 10, 13, 16, 19, 22]

# B0Map is a method
output_B0 = B0Map(image, echoes)
# T2Star is a class
output_T2Star = T2Star(image, echoes).T2Star_Nottingham()
