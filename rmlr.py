import os

f = [f for f in os.listdir('IMG') if f.startswith('left')]

# create centre 
for fl in f:
     fc = fl.replace('left','center')
     fr = fl.replace('left','right')
     if os.path.isfile('IMG/' + fc) != True:
          print('Remove : ' + fl + ' and ' + fr)
          os.remove('IMG/'+fl)
          os.remove('IMG/'+fr)


