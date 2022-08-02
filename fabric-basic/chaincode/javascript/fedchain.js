'use strict';

const stringify = require('json-stringify-deterministic');
const sortKeysRecursive = require('sort-keys-recursive');
const { Contract } = require('fabric-contract-api');

class FedChain extends Contract
{
    async CreateModelRequest(ctx, rid)
    {
        const txid = ctx.stub.getTxID();
        const modelRequestTx = {
            TransactionID: txid,
            TransactionsType: 1,
            RequesterID: rid,
            Timestamp: JSON.parse(JSON.stringify(ctx.stub.getTxTimestamp()))
        };

        await ctx.stub.putState(txid, Buffer.from(stringify(modelRequestTx)));
    }

    async AddModel(ctx, mid, modelObject, checksum)
    {
        const txid = ctx.stub.getTxID();
        const addModelTx = {
            TransactionID: txid,
            TransactionType: 2,
            ModelID: JSON.parse(mid),
            Model: JSON.parse(modelObject),
            ModelChecksum: JSON.parse(checksum),
            Timestamp: JSON.parse(JSON.stringify(ctx.stub.getTxTimestamp()))
        }

        await ctx.stub.putState(txid, Buffer.from(stringify(addModelTx)));
    }

    async ModelVerification(ctx, cid, aid, hash, quality)
    {
        const txid = ctx.stub.getTxID();
        const modelVerifyTx = {
            TransactionID: txid,
            TransactionType: 3,
            ContributorID: JSON.parse(cid),
            AggregatorID: JSON.parse(aid),
            ModelHash: JSON.parse(hash),
            ModelQuality: JSON.parse(quality),
            Timestamp: JSON.parse(JSON.stringify(ctx.stub.getTxTimestamp()))
        }

        await ctx.stub.putState(txid, Buffer.from(stringify(modelVerifyTx)));
    }

    async ReadTransaction(ctx, id) {
        const assetJSON = await ctx.stub.getState(id); // get the asset from chaincode state
        if (!assetJSON || assetJSON.length === 0) {
            throw new Error(`The transaction ${id} does not exist`);
        }
        return assetJSON.toString();
    }

    async GetAllTransactions(ctx)
    {
        const allResults = [];
        // range query with empty string for startKey and endKey does an open-ended query of all assets in the chaincode namespace.
        const iterator = await ctx.stub.getStateByRange('', '');
        let result = await iterator.next();
        while (!result.done) {
            const strValue = Buffer.from(result.value.value.toString()).toString('utf8');
            let record;
            try {
                record = JSON.parse(strValue);
            } catch (err) {
                console.log(err);
                record = strValue;
            }
            allResults.push(record);
            result = await iterator.next();
        }
        return JSON.stringify(allResults);
    }
}

module.exports = FedChain;