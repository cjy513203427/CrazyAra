#pragma once
#ifndef GRAPH_H
#define GRAPH_H
#include "node.h"
#include<vector>
class Graph
{
public:
	Graph(int capacity);
	~Graph();
    // add node to graph
	bool addNode(Node *pNode);
    // reset node
	void resetNode();
    // set adjacency matrix for directed graph
	bool setValueToMatrixForDirectedGraph(int row, int col, int val = 1);
    // set adjacency matrix for undirected graph
	bool setValueToMatrixForUndirectedGraph(int row, int col, int val = 1);
    // print adjacency matrix
	void printMatrix();

	void depthFirstTraverse(int nodeIndex);
	void breadthFirstTraverse(int nodeIndex);
	void breathFirstTraverseImpl(vector<int> preVec);

private:
	bool getValueFromMatrix(int row,int col,int &val);
	void breathFirstTraverse(int nodeIndex);

private:
    // how many nodes can a graph contain
	int m_iCapacity;
    // already added number of nodes
	int m_iNodeCount;
    // store nodes array
	Node *m_pNodeArray;
    // store adjacency matrix
	int *m_pMatrix;
};

#endif // !GRAPH_H