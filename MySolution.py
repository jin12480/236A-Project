import numpy as np
from sklearn.metrics import accuracy_score
import cvxpy as cp
### TODO: import any other packages you need for your solution


# --- Task 1 ---
class MyDecentralized:
    def __init__(self, K):
        self.K = K  # number of classes

        ### TODO: Initialize other parameters needed in your algorithm
        # examples:
        self.W = None   # shape (K, M)
        self.b = None   # shape (K,)

    def train(self, trainX, trainY):
        ''' Task 1
            TODO: train a multi-class linear classifier using LP/ILP.
                  Store learned parameters you will use in predict().
        '''
        N, M = trainX.shape
        K = self.K
        
        # Convert labels to integers if needed
        trainY = trainY.astype(int)
        
        # Map labels to 0, 1, ..., K-1 if they're not already
        unique_labels = np.unique(trainY)
        if len(unique_labels) != K or not np.array_equal(np.sort(unique_labels), np.arange(K)):
            # Create label mapping
            label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            trainY = np.array([label_map[label] for label in trainY])
            self.label_map = label_map  # Store for inverse mapping in predict
        else:
            self.label_map = None
        
        # Decision variables: weight matrix W (K x M) and bias vector b (K)
        W = cp.Variable((K, M))
        b = cp.Variable(K)
        
        # Margin parameter
        margin = 1.0
        
        # Constraints: For each training sample (x_i, y_i),
        # ensure that the correct class has the highest score with margin
        constraints = []
        for i in range(N):
            x_i = trainX[i]
            y_i = trainY[i]
            # For all classes k != y_i: w_{y_i}^T * x_i + b_{y_i} >= w_k^T * x_i + b_k + margin
            for k in range(K):
                if k != y_i:
                    constraints.append(
                        W[y_i, :] @ x_i + b[y_i] >= W[k, :] @ x_i + b[k] + margin
                    )
        
        # Objective: Minimize L2 norm of weights (regularization)
        # This helps with generalization and ensures the problem is well-posed
        objective = cp.Minimize(cp.sum_squares(W))
        
        # Solve the LP - let cvxpy choose the best available solver
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
        except:
            # Fallback to default solver if ECOS is not available
            problem.solve(verbose=False)
        
        # Check if solution was found
        if problem.status not in ["infeasible", "unbounded"] and W.value is not None:
            # Store learned parameters
            self.W = W.value
            self.b = b.value
        else:
            # If infeasible, use a fallback: one-vs-all with soft margin
            # This can happen if data is not linearly separable
            if problem.status in ["infeasible", "unbounded"]:
                print(f"Warning: LP problem status: {problem.status}. Using fallback method.")
            self._train_fallback(trainX, trainY)
    
    def _train_fallback(self, trainX, trainY):
        """Fallback training method using one-vs-all with slack variables"""
        N, M = trainX.shape
        K = self.K
        trainY = trainY.astype(int)
        
        # Map labels if needed (should already be done in train, but just in case)
        unique_labels = np.unique(trainY)
        if len(unique_labels) != K or not np.array_equal(np.sort(unique_labels), np.arange(K)):
            label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            trainY = np.array([label_map[label] for label in trainY])
        
        W = cp.Variable((K, M))
        b = cp.Variable(K)
        xi = cp.Variable(N)  # Slack variables
        
        margin = 1.0
        C = 1.0  # Regularization parameter for slack
        
        constraints = []
        for i in range(N):
            x_i = trainX[i]
            y_i = trainY[i]
            for k in range(K):
                if k != y_i:
                    constraints.append(
                        W[y_i, :] @ x_i + b[y_i] >= W[k, :] @ x_i + b[k] + margin - xi[i]
                    )
            constraints.append(xi[i] >= 0)
        
        # Minimize both weight norm and slack
        objective = cp.Minimize(cp.sum_squares(W) + C * cp.sum(xi))
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
        except:
            problem.solve(verbose=False)
        
        if problem.status not in ["infeasible", "unbounded"] and W.value is not None:
            self.W = W.value
            self.b = b.value
        else:
            # Last resort: use zero initialization
            self.W = np.zeros((K, M))
            self.b = np.zeros(K)

    def predict(self, testX):
        ''' Task 1
            TODO: predict class labels for the input data (testX) using the trained classifier
        '''
        if self.W is None or self.b is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        # Compute scores for each class: score_k = W[k] @ x + b[k]
        # testX shape: (N_test, M)
        # W shape: (K, M), b shape: (K,)
        # Scores shape: (N_test, K)
        scores = testX @ self.W.T + self.b  # Broadcasting: (N_test, M) @ (M, K) + (K,) -> (N_test, K)
        
        # Predict class with highest score (returns 0, 1, ..., K-1)
        predY = np.argmax(scores, axis=1)
        
        # Map back to original labels if needed
        if self.label_map is not None:
            inverse_map = {v: k for k, v in self.label_map.items()}
            predY = np.array([inverse_map[label] for label in predY])
        
        return predY

    def evaluate(self, testX, testY):
        predY = self.predict(testX)
        accuracy = accuracy_score(testY, predY)
        return accuracy


##########################################################################
# --- Task 2 & Task 3 ---
##########################################################################
# --- Task 2 & Task 3 ---
class MyFeatureCompression:
    def __init__(self, K):
        """
        Args:
            K (int): number of classes.
        Notes:
            You may add any state you need (e.g., a base classifier, search grids, bit candidates).
            The project does not constrain the quantizer design; document your choices and bit accounting.
        """
        self.K = K  # number of classes
        # Store a base classifier instance for reuse
        self.base_classifier = None

    def _quantize_uniform_scalar(self, X, X_train, b):
        """
        Uniform Scalar Quantization (USQ)
        
        Quantizes each feature independently using b bits per feature.
        Quantization parameters (min, max) are computed from training data only.
        
        Args:
            X: Data to quantize (N x M)
            X_train: Training data used to compute quantization parameters (N_train x M)
            b: Number of bits per feature
        
        Returns:
            X_quantized: Quantized data (N x M)
        """
        if b <= 0:
            # Zero bits: return zeros
            return np.zeros_like(X)
        
        # Compute quantization parameters from training data only
        X_min = np.min(X_train, axis=0)  # (M,)
        X_max = np.max(X_train, axis=0)  # (M,)
        
        # Handle constant features (min == max)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1.0  # Avoid division by zero
        
        # Number of quantization levels
        num_levels = 2 ** b
        
        # Quantize: map [min, max] to [0, num_levels-1]
        # Formula: q = round((x - min) / (max - min) * (num_levels - 1))
        X_normalized = (X - X_min) / X_range  # (N, M)
        X_normalized = np.clip(X_normalized, 0, 1)  # Clip to [0, 1]
        X_quantized_int = np.round(X_normalized * (num_levels - 1)).astype(int)
        X_quantized_int = np.clip(X_quantized_int, 0, num_levels - 1)
        
        # Dequantize: map back to approximate original range
        # Formula: x_hat = min + (q / (num_levels - 1)) * (max - min)
        X_quantized = X_min + (X_quantized_int / (num_levels - 1)) * X_range
        
        return X_quantized
    
    def run_centralized(self, trainX, trainY, valX, valY, testX, testY, B_tot_list):
        """
        Task 2 (Centralized compression)

        What this function should do (high level, quantizer-agnostic):
        - Assume a single centralized encoder sees all M features.
        - For each total budget in B_tot_list (measured in bits per image),
          produce a quantized representation according to your chosen quantizer,
          train a model (using train data), optionally use validation only for
          model/quantizer hyperparameter selection, and evaluate test accuracy
          on quantized test inputs.

        Args:
            trainX, trainY: training data/labels.
            valX,   valY  : validation data/labels (for model/quantizer selection only; do not touch test labels).
            testX,  testY : test data/labels (final reporting only).
            B_tot_list (Iterable[int]): list of total budgets (bits per image) to evaluate
                in the centralized setting. Example: [784, 1568, 2352, ...] corresponds to
                roughly 1/2/3/... bits per feature if you choose a uniform scalar design
                with M=784. You may implement any centralized quantizer; just ensure your
                bit accounting is clear.

        Returns:
            dict with keys:
                'B_tot'         : list[int], the budgets evaluated (bits per image)
                'test_accuracy' : list[float], test accuracy at each budget

        Notes:
            - The specification does NOT require a particular quantizer. If you use a design
              that needs data-derived parameters (e.g., ranges, codebooks), estimate them from
              training data only (no test leakage).
            - Plotting: this output is used for "accuracy vs B_tot" and to compare against Task 1.
        """
        result = {'B_tot': [], 'test_accuracy': []}
        
        N_train, M = trainX.shape
        
        # Process each budget
        for B_tot in B_tot_list:
            # Compute bits per feature for uniform scalar quantization
            # b = floor(B_tot / M) ensures we don't exceed the budget
            b = max(0, int(np.floor(B_tot / M)))
            
            # For very low budgets, we can allocate bits non-uniformly
            # For now, use uniform allocation: all features get b bits
            # This means actual budget used is b * M, which may be less than B_tot
            # but ensures we don't exceed the budget
            
            # Quantize data using training data to compute quantization parameters
            trainX_quant = self._quantize_uniform_scalar(trainX, trainX, b)
            valX_quant = self._quantize_uniform_scalar(valX, trainX, b)  # Use train params
            testX_quant = self._quantize_uniform_scalar(testX, trainX, b)  # Use train params
            
            # Train classifier on quantized training data
            clf = MyDecentralized(K=self.K)
            clf.train(trainX_quant, trainY)
            
            # Evaluate on quantized test data
            test_acc = clf.evaluate(testX_quant, testY)
            
            # Store results
            result['B_tot'].append(B_tot)
            result['test_accuracy'].append(test_acc)
            
            print(f"  Budget {B_tot} bits (b={b} bits/feature): Test Accuracy = {test_acc:.4f}")
        
        return result

    def run_decentralized_per_sensor(self, train_blocks, val_blocks, test_blocks, trainY, valY, testY, k_list):
        """
        Task 3.1 (Decentralized, fixed per-sensor budget)

        What this function should do:
        - There are 4 sensors; sensor s observes its feature block (N x d_s).
        - For each per-sensor budget k in k_list (bits per image per sensor),
          design/apply a per-sensor quantizer, concatenate the quantized blocks,
          train on quantized train, optionally use validation only for selection,
          and report test accuracy.

        Args:
            train_blocks, val_blocks, test_blocks: lists of 4 arrays, each [N x d_s],
                corresponding to the four non-overlapping quadrants (sensors).
            trainY, valY, testY: labels for train/val/test.
            k_list (Iterable[int]): list of per-sensor budgets (bits per image per sensor).
                Interpretation is up to your quantizer. A common choice is to derive a
                per-feature bit-depth b_s from k and d_s (e.g., b_s ≈ floor(k / d_s)), but
                this is not mandated; any linear-programming-consistent approach is acceptable
                as long as you document the bit accounting.

        Returns:
            dict with keys:
                'k'             : list[int], the per-sensor budgets evaluated
                'test_accuracy' : list[float], test accuracy at each k
                'b_s'           : list[tuple], optional record of per-sensor bit-depths or
                                   other allocation details per point (for reporting)

        Notes:
            - Keep train/val/test strict: use validation for selection only; do not use test
              information during training or allocation decisions.
            - Plotting: used for "accuracy vs k" and to compare with centralized at matched B_tot.
        """
        result = {'k': [], 'test_accuracy': [], 'b_s': []}
        
        num_sensors = len(train_blocks)
        assert num_sensors == 4, "Expected 4 sensor blocks"
        
        # Get feature dimensions for each sensor (should be 196 for each)
        d_s_list = [block.shape[1] for block in train_blocks]
        
        # Process each per-sensor budget
        for k in k_list:
            # Compute bits per feature for each sensor: b_s = floor(k / d_s)
            # All sensors use the same per-sensor budget k, so they all get the same b_s
            b_s_list = []
            quantized_train_blocks = []
            quantized_val_blocks = []
            quantized_test_blocks = []
            
            # Quantize each sensor block independently
            for s in range(num_sensors):
                d_s = d_s_list[s]
                b_s = max(0, int(np.floor(k / d_s)))
                b_s_list.append(b_s)
                
                # Quantize each sensor's data using its own training data for parameters
                train_block_quant = self._quantize_uniform_scalar(
                    train_blocks[s], train_blocks[s], b_s
                )
                val_block_quant = self._quantize_uniform_scalar(
                    val_blocks[s], train_blocks[s], b_s  # Use train params
                )
                test_block_quant = self._quantize_uniform_scalar(
                    test_blocks[s], train_blocks[s], b_s  # Use train params
                )
                
                quantized_train_blocks.append(train_block_quant)
                quantized_val_blocks.append(val_block_quant)
                quantized_test_blocks.append(test_block_quant)
            
            # Concatenate quantized blocks to form full feature vectors
            trainX_quant = np.concatenate(quantized_train_blocks, axis=1)
            valX_quant = np.concatenate(quantized_val_blocks, axis=1)
            testX_quant = np.concatenate(quantized_test_blocks, axis=1)
            
            # Train classifier on quantized training data
            clf = MyDecentralized(K=self.K)
            clf.train(trainX_quant, trainY)
            
            # Evaluate on quantized test data
            test_acc = clf.evaluate(testX_quant, testY)
            
            # Store results
            result['k'].append(k)
            result['test_accuracy'].append(test_acc)
            result['b_s'].append(tuple(b_s_list))  # Record bit allocation per sensor
            
            print(f"  Per-sensor budget k={k} bits (b_s={b_s_list}): Test Accuracy = {test_acc:.4f}")
        
        return result

    def _generate_bit_allocations(self, B_tot, d_s_list, max_bits=8):
        """
        Generate candidate bit allocations (b1, b2, b3, b4) for a given total budget.
        Uses a simplified strategy to keep computation tractable.
        
        Args:
            B_tot: Total budget (bits per image)
            d_s_list: List of feature dimensions for each sensor [d1, d2, d3, d4]
            max_bits: Maximum bits per feature to consider
        
        Returns:
            List of tuples (b1, b2, b3, b4) that satisfy sum(d_s * b_s) <= B_tot
        """
        num_sensors = len(d_s_list)
        allocations = []
        total_features = sum(d_s_list)
        
        # Strategy 1: Uniform allocation (all sensors get same bits)
        # b_uniform = floor(B_tot / sum(d_s))
        b_uniform = max(0, min(max_bits, int(np.floor(B_tot / total_features))))
        if sum(d * b_uniform for d in d_s_list) <= B_tot:
            allocations.append(tuple([b_uniform] * num_sensors))
        
        # Strategy 2: Try a few non-uniform allocations (limited for efficiency)
        # Only try giving one sensor +1 bit more than uniform, others get uniform
        if b_uniform < max_bits:
            for s_priority in range(num_sensors):
                b_priority = b_uniform + 1
                remaining_budget = B_tot - d_s_list[s_priority] * b_priority
                if remaining_budget >= 0:
                    other_b = max(0, min(max_bits, int(np.floor(remaining_budget / (total_features - d_s_list[s_priority])))))
                    alloc = [other_b] * num_sensors
                    alloc[s_priority] = b_priority
                    
                    # Check if allocation is valid and different
                    if sum(d * b for d, b in zip(d_s_list, alloc)) <= B_tot:
                        alloc_tuple = tuple(alloc)
                        if alloc_tuple not in allocations:
                            allocations.append(alloc_tuple)
        
        # If no allocations found, use zero allocation
        if not allocations:
            allocations.append(tuple([0] * num_sensors))
        
        return allocations
    
    def run_decentralized_total(self, train_blocks, val_blocks, test_blocks, trainY, valY, testY, B_tot_list):
        """
        Task 3.2 (Decentralized, fixed total budget)

        What this function should do:
        - Each budget B in B_tot_list is a total bit budget (bits per image) shared across 4 sensors.
        - For each B, explore one or more decentralized bit-allocation strategies across sensors
          (e.g., (b1,b2,b3,b4) if using scalar bit-depths; or any other quantizer-specific allocation),
          use validation accuracy to choose the best allocation, and report test accuracy for that choice.

        Args:
            train_blocks, val_blocks, test_blocks: lists of 4 arrays, each [N x d_s].
            trainY, valY, testY: labels for train/val/test.
            B_tot_list (Iterable[int]): list of total budgets (bits per image) to evaluate
                in the decentralized setting. For example, with scalar bit-depths one might
                constrain sum_s d_s * b_s <= B; but you may implement any decentralized quantizer,
                provided the total-bit accounting is clear and comparable.

        Returns:
            dict with keys:
                'B_tot'          : list[int], the budgets evaluated (bits per image)
                'test_accuracy'  : list[float], test accuracy at each budget
                'best_allocation': list[tuple], a record of the chosen allocation per B
                                    (e.g., (b1,b2,b3,b4) or any quantizer-specific summary)

        Notes:
            - Use validation ONLY to choose among candidate allocations or hyperparameters.
              Test is for final reporting.
            - Plotting: used for "accuracy vs B_tot" and for the centralized vs decentralized overlay.
        """
        result = {'B_tot': [], 'test_accuracy': [], 'best_allocation': []}
        
        num_sensors = len(train_blocks)
        assert num_sensors == 4, "Expected 4 sensor blocks"
        
        # Get feature dimensions for each sensor (should be 196 for each)
        d_s_list = [block.shape[1] for block in train_blocks]
        
        # Process each total budget
        for B_tot in B_tot_list:
            # Generate candidate bit allocations
            candidate_allocations = self._generate_bit_allocations(B_tot, d_s_list)
            
            best_val_acc = -1
            best_allocation = None
            best_test_acc = None
            
            # Evaluate each candidate allocation on validation set
            for allocation in candidate_allocations:
                b_s_list = list(allocation)
                
                # Quantize each sensor block with its allocated bits
                quantized_train_blocks = []
                quantized_val_blocks = []
                quantized_test_blocks = []
                
                for s in range(num_sensors):
                    b_s = b_s_list[s]
                    
                    # Quantize using training data for parameters
                    train_block_quant = self._quantize_uniform_scalar(
                        train_blocks[s], train_blocks[s], b_s
                    )
                    val_block_quant = self._quantize_uniform_scalar(
                        val_blocks[s], train_blocks[s], b_s
                    )
                    test_block_quant = self._quantize_uniform_scalar(
                        test_blocks[s], train_blocks[s], b_s
                    )
                    
                    quantized_train_blocks.append(train_block_quant)
                    quantized_val_blocks.append(val_block_quant)
                    quantized_test_blocks.append(test_block_quant)
                
                # Concatenate quantized blocks
                trainX_quant = np.concatenate(quantized_train_blocks, axis=1)
                valX_quant = np.concatenate(quantized_val_blocks, axis=1)
                testX_quant = np.concatenate(quantized_test_blocks, axis=1)
                
                # Train classifier on quantized training data
                clf = MyDecentralized(K=self.K)
                clf.train(trainX_quant, trainY)
                
                # Evaluate on validation set (for selection)
                val_acc = clf.evaluate(valX_quant, valY)
                
                # Track best allocation based on validation accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_allocation = allocation
                    # Also evaluate on test set for the best allocation
                    best_test_acc = clf.evaluate(testX_quant, testY)
            
            # Store results for the best allocation
            result['B_tot'].append(B_tot)
            result['test_accuracy'].append(best_test_acc)
            result['best_allocation'].append(best_allocation)
            
            actual_budget = sum(d * b for d, b in zip(d_s_list, best_allocation))
            print(f"  Budget {B_tot} bits (alloc={best_allocation}, actual={actual_budget}): "
                  f"Val Acc={best_val_acc:.4f}, Test Acc={best_test_acc:.4f}")
        
        return result



##########################################################################
##########################################################################
# --- Task 3.3 ---
class MyTargetAllocator:
    def __init__(self, K):
        self.K = K  # number of classes
        # TODO: add any state you need

    def minimal_bits_centralized(self, feature_compressor, trainX, trainY, valX, valY, testX, testY, alpha, B_grid):
        """
        Task 3.3 (Centralized)

        Goal:
            Given a target test accuracy α (e.g., 0.7, 0.8, 0.9), find the minimal total bit budget
            B (bits/image) so that your centralized formulation achieves test accuracy ≥ α.

        Allowed approaches (your choice, consistent with the guidelines):
            • Outer-search approach: use an outer search over candidate budgets and, for each,
              solve/evaluate your centralized formulation; pick the smallest B achieving ≥ α.
              In this case, `B_grid` provides the candidate budgets you intend to try (e.g., [784, 1568, 2352, 3136]).
            • Direct optimization approach: encode the minimal-bits objective directly in an LP/ILP/MILP
              that enforces accuracy ≥ α (as you define it) and solve for B. In this case, `B_grid` may be
              ignored or used as a search scaffold/initialization if helpful.

        Args:
            feature_compressor: an object exposing your centralized pipeline (e.g., MyFeatureCompression) if you
                choose to implement the outer-search approach. For a direct optimization approach, you may ignore it.
            trainX, trainY, valX, valY, testX, testY:
                datasets (keep train/val/test strict; no test leakage in model/quantizer selection).
            alpha (float): target test accuracy in [0,1].
            B_grid (Iterable[int]): candidate total budgets (bits/image) for the outer-search approach.
                If you implement a direct minimal-bits LP/MILP instead, you may ignore this or use it as a coarse grid.

        Returns:
            int or None:
                Minimal B (bits/image) achieving ≥ α under your centralized method; or None if not achievable
                within the search/constraints you used.

        Notes:
            • This method does not prescribe a particular quantizer or classifier; it only requires that you
              respect train/val/test separation and report bits/image clearly.
            • If multiple solutions achieve α, return the smallest B according to your method.
        """
        # Outer-search approach: search through candidate budgets
        B_grid_sorted = sorted(B_grid)
        min_B = None
        
        for B in B_grid_sorted:
            # Run centralized compression for this budget
            result = feature_compressor.run_centralized(
                trainX, trainY, valX, valY, testX, testY, [B]
            )
            
            # Check if this budget achieves the target accuracy
            if result['test_accuracy'] and result['test_accuracy'][0] >= alpha:
                min_B = B
                break
        
        return min_B

    def minimal_bits_decentralized(self, feature_compressor, train_blocks, val_blocks, test_blocks, trainY, valY, testY, alpha, B_grid):
        """
        Task 3.3 (Decentralized)

        Goal:
            Given a target test accuracy α, find the minimal total bit budget B (bits/image) and a corresponding
            decentralized allocation (e.g., per-sensor parameters such as (b1, b2, b3, b4), if that matches your design)
            so that your decentralized formulation achieves test accuracy ≥ α.

        Allowed approaches (your choice, consistent with the guidelines):
            • Outer-search approach: use an outer search over candidate budgets and, for each budget,
              search allocations/solve your decentralized formulation on train/val and evaluate on test;
              return the smallest B achieving ≥ α and its chosen allocation.
              In this case, `B_grid` provides the candidate budgets you intend to try.
            • Direct optimization approach: encode the minimal-bits objective directly in an LP/ILP/MILP
              with decentralized constraints and accuracy ≥ α; solve for B and its allocation.
              In this case, `B_grid` may be ignored or used to warm-start/coarsely bracket solutions.

        Args:
            feature_compressor: an object exposing your decentralized pipeline (e.g., MyFeatureCompression) if you
                follow the outer-search route. For a direct LP/MILP approach, you may ignore it.
            train_blocks, val_blocks, test_blocks:
                lists of 4 arrays [N × d_s], one per sensor/quadrant; keep train/val/test strict.
            trainY, valY, testY: labels.
            alpha (float): target test accuracy in [0,1].
            B_grid (Iterable[int]): candidate total budgets (bits/image) for the outer-search approach.
                If you implement a direct minimal-bits LP/MILP instead, you may ignore this or use it as a scaffold.

        Returns:
            (int or None, tuple or None):
                (minimal B, a representation of the chosen allocation at that B) if achievable; otherwise (None, None).
                The "allocation" is whatever your decentralized design uses (e.g., (b1, b2, b3, b4) for scalar bit-depths,
                or any quantizer-specific parameterization you choose to report).

        Notes:
            • This method does not prescribe how you search or solve; it only requires that you respect
              train/val/test separation and clearly report bits/image and the corresponding allocation.
            • If multiple solutions achieve α, return the one with the smallest B according to your method.
        """
        # Outer-search approach: search through candidate budgets
        B_grid_sorted = sorted(B_grid)
        min_B = None
        best_alloc = None
        
        for B in B_grid_sorted:
            # Run decentralized compression for this budget
            result = feature_compressor.run_decentralized_total(
                train_blocks, val_blocks, test_blocks,
                trainY, valY, testY, [B]
            )
            
            # Check if this budget achieves the target accuracy
            if result['test_accuracy'] and result['test_accuracy'][0] >= alpha:
                min_B = B
                best_alloc = result['best_allocation'][0] if result['best_allocation'] else None
                break
        
        return (min_B, best_alloc)
