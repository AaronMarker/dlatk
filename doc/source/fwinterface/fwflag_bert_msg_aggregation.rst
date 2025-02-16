.. _fwflag_bert_msg_aggregation:
======================
--bert_msg_aggregation
======================
Switch
======

--bert_msg_aggregation aggregation_method

Description
===========

Specify the aggregation function to apply over embedded messages to produce a single embedding of the group_id (:doc:`fwflag_c`).

Argument and Default Value
==========================

Default: mean.

Details
=======

Any method that can be called as `np.method(bertMessageVectors, axis=0)`.

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_t` , :doc:`fwflag_c`

Optional Switches:

* :doc:`fwflag_bert_model`
* :doc:`fwflag_bert_layer_aggregation`
* :doc:`fwflag_bert_layers` 

Example Commands
================

Creates a BERT feature table with messages aggregated by selecting the median.

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --add_bert --bert_model large-uncased --bert_msg_aggregation median
