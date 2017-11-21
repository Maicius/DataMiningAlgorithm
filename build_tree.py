# coding=utf-8
class tree(object):
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None


def insert(root, val):
    if root.val == None:
        root = tree(val)
    else:
        if val < root.val:
            root.left = insert(root.left, val)  # 递归地插入元素
        elif val > root.val:
            root.right = insert(root.right, val)
    return root


if __name__ == '__main__':
    data = [3, 2, 1, 5, 4]
    root = tree(None)
    for item in data:
        insert(root, item)
    print(root)
