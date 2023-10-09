#include "tree.h"
#include <iostream>
#include <queue>

Tree::Tree(int val) {
	root = new Node(val);
}

Tree::~Tree() {
	delete root;
}

void Tree::add_child(Node* parent, int val) {
	Node* child = new Node(val);
	parent->children.push_back(child);
}

void Tree::preorder_traversal(Node* node) {
	if (node == nullptr) {
		return;
	}

	std::cout << node->value << " ";

	for (Node* child : node->children) {
		preorder_traversal(child);
	}
}

void Tree::inorder_traversal(Node* node) {
	if (node == nullptr) {
		return;
	}

	if (!node->children.empty()) {
		inorder_traversal(node->children[0]);
	}

	std::cout << node->value << " ";

	for (size_t i = 1; i < node->children.size(); ++i) {
		inorder_traversal(node->children[i]);
	}
}

void Tree::postorder_traversal(Node* node) {
	if (node == nullptr) {
		return;
	}

	for (Node* child : node->children) {
		postorder_traversal(child);
	}

	std::cout << node->value << " ";
}

void Tree::bfs(Node* node) {
	if (node == nullptr) {
		return;
	}

	std::queue<Node*> q;
	q.push(node);

	while (!q.empty()) {
		Node* curr = q.front();
		q.pop();

		std::cout << curr->value << " ";

		for (Node* child : curr->children) {
			q.push(child);
		}
	}
}

void Tree::dfs(Node* node) {
	dfs_helper(node);
}

void Tree::dfs_helper(Node* node) {
	if (node == nullptr) {
		return;
	}

	std::cout << node->value << " ";

	for (Node* child : node->children) {
		dfs_helper(child);
	}
}
