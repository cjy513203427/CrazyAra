#pragma once
#ifndef TREE_H
#define TREE_H

#include "node.h"

class Tree {
public:
	Node* root;

	Tree(int val);
	~Tree();

	void add_child(Node* parent, int val);
	void preorder_traversal(Node* node);
	void inorder_traversal(Node* node);
	void postorder_traversal(Node* node);
	void bfs(Node* node);
	void dfs(Node* node);

private:
	void dfs_helper(Node* node);
};

#endif // TREE_H
