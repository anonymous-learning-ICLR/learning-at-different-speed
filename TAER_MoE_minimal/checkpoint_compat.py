"""Allow the audited entrypoint consolidation when resuming original epoch runs.

All other source changes still fail the strict checkpoint-source check.
"""

ORIGINAL_EPOCH_SOURCES = {'train_paper.py': '7ef146aa5c38176202a7856fa8f9ba066596658bb8919f501ee0f646b4e1501b',
 'paper_pipeline.py': 'e2d15068fa0de28184277d9849d572d0a75f130d4327d480d5149f4f1637597b',
 'taer_feedback.py': '1916fcd32a3a599df593ba9da145dd78184047d80bdf19c143602e07da4de04c',
 'TCNM/paper_model.py': '649113d0a16e130568d393ada49814d6236051bace01ad6422e02533782d3845',
 'TCNM/data/paper_data.py': 'd8580fa20723410494e7f84a93fa325a0af35b6e654008572e906fbf57a6eede',
 'TCNM/models_prior_unet4to4_tsi_router.py': '7b55ce1c768d1b4d5250782677bc04cc01e3278b3c64415c4af1ca1e9e6b04a3',
 'TCNM/Unet3D_merge_tiny4to4_dynamic.py': 'ee287279e1ed8fd8d5b1a7abf9df25451430c0d84324228a06c0aba70a859f7f',
 'TCNM/env_net_transformer_gphsplit.py': '299b30bc8f591306bf4ff74a7681b22d11806749cfaf30fa3f3f82ef36ce3dcf',
 'TCNM/losses.py': '6255c5cf1e44ebf0ebcc2ce63aa72fdbc338d82f571a1d795239396682c589be'}

CONSOLIDATED_SOURCES = {'train.py': 'b1d66027b18db0b2487c67c1618a338244b43a9b2705b1d77d1f307becb5e3bd',
 'paper_pipeline.py': 'd02c01bf573d95380e8e59cc40ca945731a4b86cf3318c1e131c93c8017c5016',
 'taer_feedback.py': '1916fcd32a3a599df593ba9da145dd78184047d80bdf19c143602e07da4de04c',
 'TCNM/paper_model.py': '649113d0a16e130568d393ada49814d6236051bace01ad6422e02533782d3845',
 'TCNM/data/paper_data.py': 'd8580fa20723410494e7f84a93fa325a0af35b6e654008572e906fbf57a6eede',
 'TCNM/models_prior_unet4to4_tsi_router.py': '7b55ce1c768d1b4d5250782677bc04cc01e3278b3c64415c4af1ca1e9e6b04a3',
 'TCNM/Unet3D_merge_tiny4to4_dynamic.py': 'ee287279e1ed8fd8d5b1a7abf9df25451430c0d84324228a06c0aba70a859f7f',
 'TCNM/env_net_transformer_gphsplit.py': '299b30bc8f591306bf4ff74a7681b22d11806749cfaf30fa3f3f82ef36ce3dcf',
 'TCNM/losses.py': '6255c5cf1e44ebf0ebcc2ce63aa72fdbc338d82f571a1d795239396682c589be'}


# The only training change in this migration is the feedback log label.
PREVIOUS_LOG_SOURCES = dict(CONSOLIDATED_SOURCES, **{
    'train.py': '0afb2c1fe18b77b33e0d09b76f1aa5b409b67c9f25bb8be8dcb6981d0992749a',
    'checkpoint_compat.py': '563075af2db237b52b047df8b856573b8b694d308d106c4b58e9e1ac3b5d25a4',
})


def compatible_sources(saved, current):
    if saved == current:
        return True
    return (saved in (ORIGINAL_EPOCH_SOURCES, PREVIOUS_LOG_SOURCES)
            and set(current) == set(CONSOLIDATED_SOURCES) | {'checkpoint_compat.py'}
            and all(current.get(name) == digest for name, digest in CONSOLIDATED_SOURCES.items()))
