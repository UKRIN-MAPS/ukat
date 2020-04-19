from B0Map import B0Map
from T2Star import T2Star

image = #3D/4D numpy array
echoes = [1, 4, 7, 10, 13, 16, 19, 22]

outputB0 = B0Map(image, echoes) # B0Map is a method
outputT2Star = T2Star(image, echoes).T2StarNottingham() #T2Star is a class