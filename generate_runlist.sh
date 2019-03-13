#!/bin/bash
ls SHMS_encoder_7* | sed -e 's/_/ /g' | sed -e 's/\./ /g' | awk '{print $3}' > SHMS_runlist.dat

