/*
  CrazyAra, a deep learning chess variant engine
  Copyright (C) 2018       Johannes Czech, Moritz Willig, Alena Beyer
  Copyright (C) 2019-2020  Johannes Czech

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
 * @file: searchthread.cpp
 * Created on 23.05.2019
 * @author: queensgambit
 */

#include "searchthread.h"
#ifdef TENSORRT
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include "common.h"
#endif

#include <stdlib.h>
#include <climits>
#include "util/blazeutil.h"
#include <queue>
#include <unordered_map>


size_t SearchThread::get_max_depth() const

{
    return depthMax;
}

SearchThread::SearchThread(NeuralNetAPI *netBatch, const SearchSettings* searchSettings, MapWithMutex* mapWithMutex):
    NeuralNetAPIUser(netBatch),
    rootNode(nullptr), rootState(nullptr), newState(nullptr),  // will be be set via setter methods
    newNodes(make_unique<FixedVector<Node*>>(searchSettings->batchSize)),
    newNodeSideToMove(make_unique<FixedVector<SideToMove>>(searchSettings->batchSize)),
    transpositionValues(make_unique<FixedVector<float>>(searchSettings->batchSize*2)),
    isRunning(true), mapWithMutex(mapWithMutex), searchSettings(searchSettings),
    tbHits(0), depthSum(0), depthMax(0), visitsPreSearch(0),
    terminalNodeCache(searchSettings->batchSize*2),
    reachedTablebases(false)
{
    switch (searchSettings->searchPlayerMode) {
    case MODE_SINGLE_PLAYER:
        terminalNodeCache = 1;  // TODO: Check if this is really needed
    case MODE_TWO_PLAYER: ;
    }
    searchLimits = nullptr;  // will be set by set_search_limits() every time before go()
    trajectoryBuffer.reserve(DEPTH_INIT);
    actionsBuffer.reserve(DEPTH_INIT);
}


SearchThread::SearchThread(NeuralNetAPI *netSmallBatch, NeuralNetAPI *netLargeBatch, const SearchSettings* searchSettings, MapWithMutex* mapWithMutex) :
    SearchThread(netSmallBatch, searchSettings, mapWithMutex)

{
    nnLarge = make_unique<NeuralNetAPIUser>(netLargeBatch);
}


void SearchThread::set_root_node(Node *value)
{
    rootNode = value;
    visitsPreSearch = rootNode->get_visits();
}

void SearchThread::set_root_node_large(Node *value)
{
    rootNodeLarge = value;
    visitsPreSearch = rootNodeLarge->get_visits();
    
}

void SearchThread::set_search_limits(SearchLimits *s)
{
    searchLimits = s;
}

bool SearchThread::is_running() const
{
    return isRunning;
}

void SearchThread::set_is_running(bool value)
{
    isRunning = value;
}

void SearchThread::set_reached_tablebases(bool value)
{
    reachedTablebases = value;
}

Node* SearchThread::add_new_node_to_tree(StateObj* newState, Node* parentNode, ChildIdx childIdx, NodeBackup& nodeBackup)
{
    bool transposition;
    Node* newNode = parentNode->add_new_node_to_tree(mapWithMutex, newState, childIdx, searchSettings, transposition);
    if (newNode->is_terminal()) {
        nodeBackup = NODE_TERMINAL;
        return newNode;
    }
    if (transposition) {
        const float qValue =  parentNode->get_child_node(childIdx)->get_value();
        transpositionValues->add_element(qValue);
        nodeBackup = NODE_TRANSPOSITION;
        return newNode;
    }
    nodeBackup = NODE_NEW_NODE;
    return newNode;
}

void SearchThread::stop()
{
    isRunning = false;
}

Node *SearchThread::get_root_node() const
{
    return rootNode;
}

Node *SearchThread::get_root_node_large() const
{
    return rootNodeLarge;
}

SearchLimits *SearchThread::get_search_limits() const
{
    return searchLimits;
}

void random_playout(Node* currentNode, ChildIdx& childIdx)
{
    if (currentNode->is_fully_expanded()) {
        const size_t idx = rand() % currentNode->get_number_child_nodes();
        if (currentNode->get_child_node(idx) == nullptr || !currentNode->get_child_node(idx)->is_playout_node()) {
            childIdx = idx;
            return;
        }
        if (currentNode->get_child_node(idx)->get_node_type() == UNSOLVED) {
            childIdx = idx;
            return;
        }
        childIdx = uint16_t(-1);
    }
    else {
        childIdx = min(size_t(currentNode->get_no_visit_idx()), currentNode->get_number_child_nodes()-1);
        currentNode->increment_no_visit_idx();
        return;
    }
}

Node* SearchThread::get_starting_node(Node* currentNode, NodeDescription& description, ChildIdx& childIdx)
{
    size_t depth = get_random_depth();
    for (uint curDepth = 0; curDepth < depth; ++curDepth) {
        currentNode->lock();
        childIdx = get_best_action_index(currentNode, true, searchSettings);
        Node* nextNode = currentNode->get_child_node(childIdx);
        if (nextNode == nullptr || !nextNode->is_playout_node() || nextNode->get_visits() < searchSettings->epsilonGreedyCounter || nextNode->get_node_type() != UNSOLVED) {
            currentNode->unlock();
            break;
        }
        currentNode->unlock();
        actionsBuffer.emplace_back(currentNode->get_action(childIdx));
        currentNode = nextNode;
        ++description.depth;
    }
    return currentNode;
}

Node* SearchThread::get_new_child_to_evaluate(NodeDescription& description, Node* rootNode)
{
    description.depth = 0;
    Node* currentNode = rootNode;
    Node* nextNode;

    ChildIdx childIdx = uint16_t(-1);
    if (searchSettings->epsilonGreedyCounter && rootNode->is_playout_node() && rand() % searchSettings->epsilonGreedyCounter == 0) {
        currentNode = get_starting_node(currentNode, description, childIdx);
        currentNode->lock();
        random_playout(currentNode, childIdx);
        currentNode->unlock();
    }
    else if (searchSettings->epsilonChecksCounter && rootNode->is_playout_node() && rand() % searchSettings->epsilonChecksCounter == 0) {
        currentNode = get_starting_node(currentNode, description, childIdx);
        currentNode->lock();
        childIdx = select_enhanced_move(currentNode);
        if (childIdx ==  uint16_t(-1)) {
            random_playout(currentNode, childIdx);
        }
        currentNode->unlock();
    }

    while (true) {
        currentNode->lock();
        if (childIdx == uint16_t(-1)) {
            childIdx = currentNode->select_child_node(searchSettings);
        }
        currentNode->apply_virtual_loss_to_child(childIdx, searchSettings);
        trajectoryBuffer.emplace_back(NodeAndIdx(currentNode, childIdx));

        nextNode = currentNode->get_child_node(childIdx);
        description.depth++;
        if (nextNode == nullptr) {
#ifdef MCTS_STORE_STATES
            StateObj* newState = currentNode->get_state()->clone();
#else
            newState = unique_ptr<StateObj>(rootState->clone());
            assert(actionsBuffer.size() == description.depth-1);
            for (Action action : actionsBuffer) {
                newState->do_action(action);
            }
#endif
            newState->do_action(currentNode->get_action(childIdx));
            currentNode->increment_no_visit_idx();
#ifdef MCTS_STORE_STATES
            nextNode = add_new_node_to_tree(newState, currentNode, childIdx, description.type);
#else
            nextNode = add_new_node_to_tree(newState.get(), currentNode, childIdx, description.type);
#endif
            currentNode->unlock();

            if (description.type == NODE_NEW_NODE) {
#ifdef SEARCH_UCT
                Node* nextNode = currentNode->get_child_node(childIdx);
                nextNode->set_value(newState->random_rollout());
                nextNode->enable_has_nn_results();
                if (searchSettings->useTranspositionTable && !nextNode->is_terminal()) {
                    mapWithMutex->mtx.lock();
                    mapWithMutex->hashTable.insert({nextNode->hash_key(), nextNode});
                    mapWithMutex->mtx.unlock();
                }
#else
                // fill a new board in the input_planes vector
                // we shift the index by nbNNInputValues each time
                newState->get_state_planes(true, inputPlanes + newNodes->size() * net->get_nb_input_values_total(), net->get_version());
                // save a reference newly created list in the temporary list for node creation
                // it will later be updated with the evaluation of the NN
                newNodeSideToMove->add_element(newState->side_to_move());
#endif
            }
            return nextNode;
        }
        if (nextNode->is_terminal()) {
            description.type = NODE_TERMINAL;
            currentNode->unlock();
            return nextNode;
        }
        if (!nextNode->has_nn_results()) {
            description.type = NODE_COLLISION;
            currentNode->unlock();
            return nextNode;
        }
        if (nextNode->is_transposition()) {
            nextNode->lock();
            const uint_fast32_t transposVisits = currentNode->get_real_visits(childIdx);
            const double transposQValue = currentNode->get_transposition_q_value(searchSettings, childIdx, transposVisits);

            if (nextNode->is_transposition_return(transposQValue)) {
                const float backupValue = get_transposition_backup_value(transposVisits, transposQValue, nextNode->get_value());
                nextNode->unlock();
                description.type = NODE_TRANSPOSITION;
                transpositionValues->add_element(backupValue);
                currentNode->unlock();
                return nextNode;
            }
            nextNode->unlock();
        }
        currentNode->unlock();
#ifndef MCTS_STORE_STATES
        actionsBuffer.emplace_back(currentNode->get_action(childIdx));
#endif
        currentNode = nextNode;
        childIdx = uint16_t(-1);
    }
}

void SearchThread::set_root_state(StateObj* value)
{
    rootState = value;
}

size_t SearchThread::get_tb_hits() const
{
    return tbHits;
}

void SearchThread::reset_stats()
{
    tbHits = 0;
    depthMax = 0;
    depthSum = 0;
}

void fill_nn_results(size_t batchIdx, bool isPolicyMap, const float* valueOutputs, const float* probOutputs, const float* auxiliaryOutputs, Node *node, size_t& tbHits, bool mirrorPolicy, const SearchSettings* searchSettings, bool isRootNodeTB)
{
    node->set_probabilities_for_moves(get_policy_data_batch(batchIdx, probOutputs, isPolicyMap), mirrorPolicy);
    node_post_process_policy(node, searchSettings->nodePolicyTemperature, searchSettings);
    node_assign_value(node, valueOutputs, tbHits, batchIdx, isRootNodeTB);
#ifdef MCTS_STORE_STATES
    node->set_auxiliary_outputs(get_auxiliary_data_batch(batchIdx, auxiliaryOutputs));
#endif
    node->enable_has_nn_results();
}

void SearchThread::set_nn_results_to_child_nodes()
{
    size_t batchIdx = 0;
    for (auto node: *newNodes) {
        fill_nn_results(batchIdx, net->is_policy_map(), valueOutputs, probOutputs, auxiliaryOutputs, node,
                        tbHits, rootState->mirror_policy(newNodeSideToMove->get_element(batchIdx)),
                        searchSettings, rootNode->is_tablebase());
        ++batchIdx;
    }
}

void SearchThread::backup_value_outputs()
{
    backup_values(*newNodes, newTrajectories);
    newNodeSideToMove->reset_idx();
    backup_values(transpositionValues.get(), transpositionTrajectories);
}

void SearchThread::backup_collisions() {
    for (size_t idx = 0; idx < collisionTrajectories.size(); ++idx) {
        backup_collision(searchSettings, collisionTrajectories[idx]);
    }
    collisionTrajectories.clear();
}

bool SearchThread::nodes_limits_ok()
{
    return (searchLimits->nodes == 0 || (rootNode->get_node_count() < searchLimits->nodes)) &&
            (searchLimits->simulations == 0 || (rootNode->get_visits() < searchLimits->simulations)) &&
            (searchLimits->nodesLimit == 0 || (rootNode->get_node_count() < searchLimits->nodesLimit));
}

bool SearchThread::is_root_node_unsolved()
{
#ifdef MCTS_TB_SUPPORT
    return is_unsolved_or_tablebase(rootNode->get_node_type());
#else
    return rootNode->get_node_type() == UNSOLVED;
#endif
}

size_t SearchThread::get_avg_depth()
{
    return size_t(double(depthSum) / (rootNode->get_visits() - visitsPreSearch) + 0.5);
}

void SearchThread::create_mini_batch(Node* rootNode)
{
    // select nodes to add to the mini-batch
    NodeDescription description;
    size_t numTerminalNodes = 0;

    while (!newNodes->is_full() &&
           collisionTrajectories.size() != searchSettings->batchSize &&
           !transpositionValues->is_full() &&
           numTerminalNodes < terminalNodeCache) {
        simulation_puct(rootNode, numTerminalNodes);
    }
}

void SearchThread::simulation_puct(Node* rootNode, size_t &numTerminalNodes){
    NodeDescription description;
    trajectoryBuffer.clear();
    actionsBuffer.clear();
    Node* newNode = get_new_child_to_evaluate(description, rootNode);
    depthSum += description.depth;
    depthMax = max(depthMax, description.depth);

    if(description.type == NODE_TERMINAL) {
        ++numTerminalNodes;
        backup_value<true>(newNode->get_value(), searchSettings, trajectoryBuffer, searchSettings->mctsSolver);
    }
    else if (description.type == NODE_COLLISION) {
        // store a pointer to the collision node in order to revert the virtual loss of the forward propagation
        collisionTrajectories.emplace_back(trajectoryBuffer);
    }
    else if (description.type == NODE_TRANSPOSITION) {
        transpositionTrajectories.emplace_back(trajectoryBuffer);
    }
    else {  // NODE_NEW_NODE
        newNodes->add_element(newNode);
        newTrajectories.emplace_back(trajectoryBuffer);
    }
}

void SearchThread::thread_iteration()
{
    if(searchSettings->useMPVMCTS){
        mpv_mcts(5, 3);
    }else{
        create_mini_batch(rootNode);
        // create_mini_batch(rootNodeLarge);
#ifndef SEARCH_UCT
        if (newNodes->size() != 0) {
            net->predict(inputPlanes, valueOutputs, probOutputs, auxiliaryOutputs);
            set_nn_results_to_child_nodes();
            
        }
#endif
        backup_value_outputs();
        backup_collisions();
    }

}

void run_search_thread(SearchThread* t)
{
    t->set_is_running(true);
    t->reset_stats();
    while(t->is_running() && t->nodes_limits_ok() && t->is_root_node_unsolved()) {
        t->thread_iteration();
    }
    t->set_is_running(false);
}

void SearchThread::backup_values(FixedVector<Node*>& nodes, vector<Trajectory>& trajectories) {
    for (size_t idx = 0; idx < nodes.size(); ++idx) {
        Node* node = nodes.get_element(idx);
#ifdef MCTS_TB_SUPPORT
        const bool solveForTerminal = searchSettings->mctsSolver && node->is_tablebase();
        backup_value<false>(node->get_value(), searchSettings, trajectories[idx], solveForTerminal);
#else
        backup_value<false>(node->get_value(), searSearch_TypechSettings, trajectories[idx], false);
#endif
    }
    nodes.reset_idx();
    trajectories.clear();
}

void SearchThread::backup_values(FixedVector<float>* values, vector<Trajectory>& trajectories) {
    for (size_t idx = 0; idx < values->size(); ++idx) {
        const float value = values->get_element(idx);
        backup_value<true>(value, searchSettings, trajectories[idx], false);
    }
    values->reset_idx();
    trajectories.clear();
}

void SearchThread::backup_values(FixedVector<Node*> *nodes) {
    for (size_t idx = 0; idx < nodes->size(); ++idx) {
        Node* curNode = nodes->get_element(idx);
        const float value = curNode->get_value();
        backup_value<true>(value, searchSettings, curNode->get_parent_node(), false, curNode->get_parent_node_idx());
    }
    nodes->reset_idx();
}

ChildIdx SearchThread::select_enhanced_move(Node* currentNode) const {
    if (currentNode->is_playout_node() && !currentNode->was_inspected() && !currentNode->is_terminal()) {

        // iterate over the current state
        unique_ptr<StateObj> pos = unique_ptr<StateObj>(rootState->clone());
        for (Action action : actionsBuffer) {
            pos->do_action(action);
        }

        // make sure a check has been explored at least once
        for (size_t childIdx = currentNode->get_no_visit_idx(); childIdx < currentNode->get_number_child_nodes(); ++childIdx) {
            if (pos->gives_check(currentNode->get_action(childIdx))) {
                for (size_t idx = currentNode->get_no_visit_idx(); idx < childIdx+1; ++idx) {
                    currentNode->increment_no_visit_idx();
                }
                return childIdx;
            }
        }
        // a full loop has been done
        currentNode->set_as_inspected();
    }
    return uint16_t(-1);
}

void SearchThread::mpv_mcts(size_t b_Small, size_t b_Large){
    vector<int> list = randomly_select(b_Small, b_Small + b_Large, 4);
    NodeDescription description;
    for (int i=0; i<b_Small + b_Large; i++){
        actionsBuffer.clear();
        trajectoryBuffer.clear();
        // if i in list
        bool found = false;
        for (const int &element : list) {
            if (element == i) {
                found = true;
                break;
            }
        }

        if(found){
            // S_leaf = SelectUnevaluatedLeafStateByPUCT(T_S)
            select_unevaluated_leafState_puct(rootNode);
            net->predict(inputPlanes, valueOutputs, probOutputs, auxiliaryOutputs);
            // (p, v) = f_S (s_leaf)
            set_nn_results_to_child_nodes();
            // Update(T_S, s_leaf, (p, v))
            backup_value_outputs();
            backup_collisions();
        }else{
            // S_leaf = SelectUnevaluatedLeafStateByPriority(T_L)
            select_unevaluated_leafState_priority(rootNodeLarge);
            nnLarge->get_net()->predict(inputPlanes, valueOutputs, probOutputs, auxiliaryOutputs);
            if(rootNode->get_real_visits() == 0)
                // S_leaf = SelectUnevaluatedLeafStateByPUCT(T_L)
                select_unevaluated_leafState_puct(rootNodeLarge);
            // (p, v) = f_L (s_leaf)
            set_nn_results_to_child_nodes();
            // Update(T_L, s_leaf, (p, v))
            update(description);
        }
    }
}

vector<int> SearchThread::randomly_select(int lowerbound, int upperbound, int num_selections){  
    // seed
    random_device rd;
    mt19937 gen(rd());
    // generate random num
    uniform_int_distribution<> dis(lowerbound, upperbound);

    vector<int> selections;
    for (int i = 0; i < num_selections; ++i) {
        selections.push_back(dis(gen));
    }


    return selections;
}  

void SearchThread::select_unevaluated_leafState_puct(Node* rootNode){
    size_t num_loop = 0;
    simulation_puct(rootNode, num_loop);
    simulation_puct(rootNodeLarge, num_loop);
}

void SearchThread::select_unevaluated_leafState_priority(Node* rootNode){
    // Priority means higher visit counts(based on small tree)
    // For each node have potential nodes, choose important nodes which has the most qvalues. The best move has the most visits. Subsequent nodes and opponent move are also important. Future moves take into account.
    // Get final rootNode
    // Todo: Exclude nodes whose qVal are -1, after excluding, rootNodeMap's size is 0?
    // get key through hashKey. Need for loop map.
    

    // cout<< "--------------------------t->iterate_all_nodes_bfs(rootNodeLargeBFS)--------------------------" << endl;
    std::multimap<Key, std::pair<Node*, ChildIdx>> rootNodeLargeMap = this->iterate_all_nodes_bfs(rootNodeLarge);
    // cout<<"Size of rootNodeLargeMap = "<< rootNodeLargeMap.size()<<endl;

    // store visits from small tree and node from large tree
    std::multimap <unsigned int, std::pair<Node*, ChildIdx>, std::greater<unsigned int>> combinedRootNodeLargeMap;

//    std::multimap<std::pair<Key, unsigned int>, Node*> rootNodeLargeDoublekeyMap = this->doublekey_map(rootNodeLargeMap);

    //auto firstElementIterator = sortedRootNodeMapping.begin();
    auto firstElementIterator = rootNodeLargeMap.begin();
    // std::cout << "First key of the first element: " << firstKey << std::endl;

    // check matched key
    for (auto it = rootNodeLargeMap.begin(); it != rootNodeLargeMap.end(); ++it) {
        Key currentKey = it->first;

        mapWithMutex->mtx.lock();
        HashMap::const_iterator itt = mapWithMutex->hashTable.find(currentKey);
        if (itt != mapWithMutex->hashTable.end()) {
            shared_ptr<Node> matchedNode = itt->second.lock();
            if (matchedNode->get_node_data() != nullptr) {
                combinedRootNodeLargeMap.emplace(matchedNode->get_visits(), std::make_pair(it->second.first, it->second.second));
            }
        }

        mapWithMutex->mtx.unlock();


    }


    if (combinedRootNodeLargeMap.size() > 0 && combinedRootNodeLargeMap.begin()->second.first!=nullptr) {
        // retrieve the parent node
        Node * parentNode = combinedRootNodeLargeMap.begin()->second.first;
        const ChildIdx childIdx =  combinedRootNodeLargeMap.begin()->second.second;
        // create a new node
        // we need to clone the state to avoid double free problem for MCTS_STORE_STATES
        StateObj* state = parentNode->get_state()->clone();
        state->do_action(parentNode->get_action(childIdx));
        shared_ptr<Node> newNode = make_shared<Node>(state, searchSettings, parentNode, childIdx);
        state->get_state_planes(true, inputPlanes + newNodes->size() * net->get_nb_input_values_total(), net->get_version());
        newNodeSideToMove->add_element(state->side_to_move());
        state->undo_action(parentNode->get_action(childIdx));
        // connect the Node to the parent
        parentNode->fully_expand_node();
        parentNode->connect_child_node(newNode, childIdx);
        newNodes->add_element(newNode.get());

    }


}


std::multimap<Key, std::pair<Node*, ChildIdx>> SearchThread::iterate_all_nodes_bfs(Node* node)
{
    // a queue for traverse
    std::queue<Node*> q;
    // key: number of the visits, value: node pointer which wants to be evaluated
    std::multimap<Key, std::pair<Node*, ChildIdx>> leafNodesMap;

    q.push(node);

    vector<Key> keys;
    while (!q.empty()) {
        Node* curNode = q.front();

        q.pop();


        NodeData* curData = curNode->get_node_data();

        // std::cout << "curNode->get_value_sum(): " << curNode->get_value_sum() << endl;

        //if(curData == nullptr || curNode->is_sorted() == false){
        if (curData == nullptr) {
            continue;
        }

        StateObj* newState = curNode->get_state()->clone();

        for (ChildIdx idx = curData->noVisitIdx; idx < curNode->get_number_child_nodes(); idx++) {
            Action action = curNode->get_action(idx);
            if(action == MOVE_NULL){
                continue;
            }
            newState->do_action(action);
            keys.emplace_back(newState->hash_key());
            leafNodesMap.emplace(newState->hash_key(), std::make_pair(curNode, idx));
            newState->undo_action(action);

        }

        // If a node is leaf node
        // There are many identical numbers.
        //std::cout << "curData->noVisitIdx: " << curData->noVisitIdx << endl;

        for (size_t idx = 0; idx < curData->childNodes.size(); ++idx) {
            if (curData->childNodes[idx] != nullptr) {
                q.push(curData->childNodes[idx].get());
            }
        }

    }
    return leafNodesMap;
}

std::multimap<std::pair<Key, unsigned int>, Node*> SearchThread::doublekey_map(std::multimap<unsigned int, Node*, std::greater<unsigned int>> treeMap) {
    std::multimap<std::pair<Key, unsigned int>, Node*> rootNodeMapping;

    for (const auto& pair : treeMap) {
        // std::cout << pair.first << ": " << pair.second->hash_key() << std::endl;
        rootNodeMapping.emplace(std::make_pair(pair.second->hash_key(), pair.first), pair.second);

    }

    return rootNodeMapping;
}

std::multimap<std::pair<Key, unsigned int>, Node*> SearchThread::create_mapping_for_small_large_tree(std::multimap<unsigned int, Node*, std::greater<unsigned int>> smallTreeMap, std::multimap<unsigned int, Node*, std::greater<unsigned int>> largeTreeMap){
    /*
        traverse smallTreeMap and largeTreeMap
        retrieve "hash_key" from nodes (value of map) as a new key of map as first key
        "visists" as the second key
                value keeps unchanged
        return a new map
        use MapWithMutex
     */
    mapWithMutex->mtx.lock();
    std::multimap<std::pair<Key, unsigned int>, Node*> rootNodeMapping;

    smallTreeMap.insert(largeTreeMap.begin(), largeTreeMap.end());
    for (const auto& pair : smallTreeMap) {
        // std::cout << pair.first << ": " << pair.second->hash_key() << std::endl;
        rootNodeMapping.emplace(std::make_pair(pair.second->hash_key(), pair.first), pair.second);

    }

    mapWithMutex->mtx.unlock();
    return rootNodeMapping;
}

void SearchThread::update(NodeDescription& description){
    // backup_value_outputs();
    // backup_collisions();

     backup_values(newNodes.get());
//        newNodes->reset_idx();
}

void node_assign_value(Node *node, const float* valueOutputs, size_t& tbHits, size_t batchIdx, bool isRootNodeTB)
{
#ifdef MCTS_TB_SUPPORT
    if (node->is_tablebase()) {
        ++tbHits;
        // TODO: Improvement the value assignment for table bases
        if (node->get_value() != 0 && isRootNodeTB) {
            // use the average of the TB entry and NN eval for non-draws
            node->set_value((valueOutputs[batchIdx] + node->get_value()) * 0.5f);
        }
        return;
    }
#endif
    node->set_value(valueOutputs[batchIdx]);
}

void node_post_process_policy(Node *node, float temperature, const SearchSettings* searchSettings)
{
    node->enhance_moves(searchSettings);
    node->apply_temperature_to_prior_policy(temperature);
}

size_t get_random_depth()
{
    const int randInt = rand() % 100 + 1;
    return std::ceil(-std::log2(1 - randInt / 100.0) - 1);
}

