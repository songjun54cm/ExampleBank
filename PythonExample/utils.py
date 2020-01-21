# display function, display array and matrix
def display(values, show=True):
    print('type:%s\n'
          'shape: %s' % (values[0].dtype, values[0].shape))
    if show:
        for i in range(3):
            print('example%s\n%s' % (i, values[i]))
