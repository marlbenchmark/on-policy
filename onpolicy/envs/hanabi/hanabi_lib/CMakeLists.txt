add_library (hanabi hanabi_card.cc hanabi_game.cc hanabi_hand.cc hanabi_history_item.cc hanabi_move.cc hanabi_observation.cc hanabi_state.cc util.cc canonical_encoders.cc)
target_include_directories(hanabi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
