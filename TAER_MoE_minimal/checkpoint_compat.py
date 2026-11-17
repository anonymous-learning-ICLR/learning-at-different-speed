"""Resume the supplied checkpoint after consolidating its training entry point."""

ORIGINAL_SOURCES = {'train_paper.py': '494e14b202df46db3b714233c844dc2fdecd797c03a8902d081fe898a4896819', 'paper_pipeline.py': '845c5239ca032a38751d5002aee7d775532cee254e85cecd960918bfbe40c2ac', 'training_checkpoints.py': '8a620a66d35a25860fa2fe426a17be9e56d1699acf7f9092d11e79ac5e6d19b7', 'taer_feedback.py': '1916fcd32a3a599df593ba9da145dd78184047d80bdf19c143602e07da4de04c', 'TCNM/paper_model.py': '649113d0a16e130568d393ada49814d6236051bace01ad6422e02533782d3845', 'TCNM/data/paper_data.py': 'd8580fa20723410494e7f84a93fa325a0af35b6e654008572e906fbf57a6eede', 'TCNM/models_prior_unet4to4_tsi_router.py': '7b55ce1c768d1b4d5250782677bc04cc01e3278b3c64415c4af1ca1e9e6b04a3', 'TCNM/Unet3D_merge_tiny4to4_dynamic.py': 'ee287279e1ed8fd8d5b1a7abf9df25451430c0d84324228a06c0aba70a859f7f', 'TCNM/env_net_transformer_gphsplit.py': '299b30bc8f591306bf4ff74a7681b22d11806749cfaf30fa3f3f82ef36ce3dcf', 'TCNM/losses.py': '6255c5cf1e44ebf0ebcc2ce63aa72fdbc338d82f571a1d795239396682c589be'}

RELEASE_SOURCES = {'train.py': '185a3ca2630da8fb386c02483ff555784496fc952f7d928df5be305bb61d9bee', 'paper_pipeline.py': '96e6a6390cedfe76e7abe8fc7935b6af9a9f3e3f74ad3004cbab91418e63791a', 'training_checkpoints.py': '7a2481a9c816c9e17aaa4832394e566d97965fae6b1228c94ee39a72efaefea8', 'taer_feedback.py': '1916fcd32a3a599df593ba9da145dd78184047d80bdf19c143602e07da4de04c', 'TCNM/paper_model.py': '649113d0a16e130568d393ada49814d6236051bace01ad6422e02533782d3845', 'TCNM/data/paper_data.py': 'd8580fa20723410494e7f84a93fa325a0af35b6e654008572e906fbf57a6eede', 'TCNM/models_prior_unet4to4_tsi_router.py': '7b55ce1c768d1b4d5250782677bc04cc01e3278b3c64415c4af1ca1e9e6b04a3', 'TCNM/Unet3D_merge_tiny4to4_dynamic.py': 'ee287279e1ed8fd8d5b1a7abf9df25451430c0d84324228a06c0aba70a859f7f', 'TCNM/env_net_transformer_gphsplit.py': '299b30bc8f591306bf4ff74a7681b22d11806749cfaf30fa3f3f82ef36ce3dcf', 'TCNM/losses.py': '6255c5cf1e44ebf0ebcc2ce63aa72fdbc338d82f571a1d795239396682c589be'}


def compatible_sources(saved, current):
    if saved == current:
        return True
    return (saved == ORIGINAL_SOURCES
            and set(current) == set(RELEASE_SOURCES) | {'checkpoint_compat.py'}
            and all(current.get(name) == digest for name, digest in RELEASE_SOURCES.items()))
