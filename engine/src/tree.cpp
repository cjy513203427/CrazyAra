#include"tree.h"
#include<iostream>
using namespace std;
Tree::Tree()
{
	m_pRoot = new Node(0,0,NULL,NULL,NULL);
};

Tree::~Tree()
{
	m_pRoot->delete_node();
}

Node *Tree::search_node(int nodeIndex)
{
	return m_pRoot->search_node(nodeIndex);
}
//nodeIndex is the index of father node
bool Tree::add_node(int nodeIndex, int direction, Node* pNode)
{
	Node *temp = search_node(nodeIndex);
	if (temp == NULL)
	{
		return false;
	}

	Node *node = new Node();
	if (node == NULL)
	{
		return false;
	}
	node->index = pNode->index;
	node->data = pNode->data;
	node->pParent = temp;

	if (direction == 0)
	{
		temp->pLChild = node;
	}

	if (direction == 1)
	{
		temp->pRChild = node;
	}

	return true;
}

bool Tree::delete_node(int nodeIndex, Node* pNode)
{
	Node *temp = search_node(nodeIndex);
	if (temp == NULL)
	{
		return false;
	}

	if (pNode != NULL)
	{
		pNode->data = temp->data;
	}

	temp->delete_node();
	return true;
}

void Tree::preorder_traversal()
{
	m_pRoot->preorder_traversal();
}

void Tree::inorder_traversal()
{
	m_pRoot->inorder_traversal();
}

void Tree::postorder_traversal()
{
	m_pRoot->postorder_traversal();
}