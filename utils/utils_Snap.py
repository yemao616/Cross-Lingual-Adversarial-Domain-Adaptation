from anytree import Node, RenderTree
from anytree.search import findall_by_attr
from anytree.walker import Walker
import pandas as pd
import json


def parsing_trees(code):
    # load json code
    json_code = json.loads(code)

    # Initialize head node of the code.
    head = Node(["1", json_code['type']], token=json_code['type'], value_node=False)

    # Recursively construct AST tree.
    for child_order in json_code['childrenOrder']:
        converting_trees(json_code['children'][child_order], head, "1" + str(int(child_order) + 1))

    return head


def converting_trees(json_code, parent_node, order):
    """
    Converting single json code to an AST tree.
    Parameters
    ----------
    json_code:      tuple, json object. The json object of a snap program.
    parent_node:    Node, Parent node, used as parent in this recursion.
    order:          str, Storing the current order of this node with other children.

    """
    # Record current node, children order.
    '''
    if 'value' in json_code:
        node = Node([order,json_code['type']+":"+json_code['value']], parent=parent_node, order=order)
    else:
        node = Node([order,json_code['type']], parent=parent_node, order=order)
    '''
    token = json_code['type']
    node = Node([order, token], parent=parent_node, order=order, token=token, value_node=False)

    # If current json part doesn't have a child, return.
    if not json_code.get('childrenOrder'):
        if json_code.get('value') and token not in ['stage', 'sprite']:
            value_order = order + '1'
            value_node = Node([value_order, json_code['value']], parent=node, order=value_order,
                              token=json_code['value'], value_node=True)
        return

    # Recursion for every child under current node.
    for child_order in json_code['childrenOrder']:
        converting_trees(json_code['children'][child_order], node, order + str(int(child_order) + 1))


def get_sequences(node, sequence):
    token, children = node.token, node.children
    sequence.append(token)

    for child in children:
        get_sequences(child, sequence)

    if token in ['script', 'customBlock']:
        sequence.append('End')


def get_blocks(node, block_seq):

    token = node.token
    children = node.children
    if token in ['snapshot']:
        block_seq.append(node)  # add snapshot root and directly append children
        for snapshot_child in children:
            snapshot_child.parent = None
            block_seq.append(snapshot_child)
            get_blocks(snapshot_child, block_seq)

    elif token in ['customBlock']:
        for customBlock_child in children:
            if customBlock_child.token not in ['script'] and not customBlock_child.value_node:
                customBlock_child.parent = None  # separate customBlock and its parent node
                block_seq.append(customBlock_child)
            get_blocks(customBlock_child, block_seq)

        block_seq.append(Node('End', token='End'))

    elif token in ['script']:
        for script_child in children:
            # separate script and its parent node since script will be reached from top node
            script_child.parent = None
            block_seq.append(script_child)
            get_blocks(script_child, block_seq)

        block_seq.append(Node('End', token='End'))

    else:
        # en-route node, do not add to block_seq
        for child_node in children:
            get_blocks(child_node, block_seq)

