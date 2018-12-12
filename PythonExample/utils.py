# display function, desplay array and matrix
def display(alist, show = True):
    print('type:%s\nshape: %s' %(alist[0].dtype,alist[0].shape))
    if show:
        for i in range(3):
            print('example%s\n%s' %(i,alist[i]))