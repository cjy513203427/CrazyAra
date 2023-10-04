#pragma once
#ifndef  TREE_H
#define TREE_H
#include "node.h"
class Tree
{
public:
	Tree();
	~Tree();
	Node *search_node(int nodeIndex);
	bool add_node(int nodeIndex, int direction, Node* pNode);
	bool delete_node(int nodeIndex, Node* pNode);
	void preorder_traversal();
	void inorder_traversal();
	void postorder_traversal();
private:
	Node * m_pRoot;
};

#endif // ! TREE_H