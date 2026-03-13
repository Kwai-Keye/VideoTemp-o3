import os
from typing import List, Optional, Tuple

import json

from swift.llm import Messages
from swift.llm.template.utils import ContextType
from swift.plugin.loss_scale.loss_scale import LossScale, loss_scale_map

class LastTwoRoundsLossScale(LossScale):
    """Compute loss only on the last two assistant responses; all other rounds have a weight of 0"""

    def get_loss_scale(self, context: str, context_type: ContextType, is_last_round: bool, **kwargs):
        # Only process the response part (i.e., assistant's outputs)
        if context_type == ContextType.RESPONSE:
            assistant_turns = kwargs.get('assistant_turns', [])
            current_turn_idx = kwargs.get('current_turn_idx', -1)
            
            # Check whether the current turn is one of the last two
            if len(assistant_turns) <= 2:
                # If there are less than or equal to two assistant turns, compute loss for all
                return [context], [1.0]
            else:
                # Only the last two rounds are considered for loss (indices -1 and -2)
                if current_turn_idx in assistant_turns[-2:]:
                    return [context], [1.0]
                else:
                    return [context], [0.0]
        # Non-response parts (e.g., user inputs) do not compute loss
        return super().get_loss_scale(context, context_type, is_last_round)

    def __call__(self, context_list: List[str], context_types: List[ContextType], messages: Messages,** kwargs) -> Tuple[List[str], List[float]]:
        res_context_list = []
        res_loss_scale = []
        # Record the current assistant turn count
        assistant_turn_count = 0
        # Extract all assistant turn indices (used to determine if it's one of the last two turns)
        assistant_turns = [i for i, msg in enumerate(messages) if msg['role'] == 'assistant']
        kwargs['assistant_turns'] = assistant_turns

        for context, context_type in zip(context_list, context_types):
            # Only count assistant response turns
            if context_type == ContextType.RESPONSE:
                # Pass the current turn index to get_loss_scale
                current_turn_idx = assistant_turns[assistant_turn_count] if assistant_turn_count < len(assistant_turns) else -1
                kwargs['current_turn_idx'] = current_turn_idx
                assistant_turn_count += 1

            # Call get_loss_scale to compute weights
            new_context, loss_scale = self.get_loss_scale(
                context, context_type, 
                is_last_round=(assistant_turn_count == len(assistant_turns)),  # Whether it's the last round (for auxiliary judgment)
                **kwargs
            )
            res_context_list.extend(new_context)
            res_loss_scale.extend(loss_scale)
        return res_context_list, res_loss_scale

loss_scale_map['last_two_rounds'] = LastTwoRoundsLossScale