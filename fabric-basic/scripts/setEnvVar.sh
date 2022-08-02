#/bin/sh

setEnvVar()
{
    export CORE_PEER_TLS_ENABLED=true
    export CORE_PEER_LOCALMSPID="Org1MSP"
    export CORE_PEER_MSPCONFIGPATH=${PWD}/crypto/peerOrganizations/org1.fedchain/users/Admin@org1.fedchain/msp
    case $1 in
        1)
            export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/crypto/peerOrganizations/org1.fedchain/peers/peer1.org1.fedchain/tls/ca.crt
            export CORE_PEER_ADDRESS=peer1.org1.fedchain:7051
            ;;
        2)
            export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/crypto/peerOrganizations/org1.fedchain/peers/peer2.org1.fedchain/tls/ca.crt
            export CORE_PEER_ADDRESS=peer2.org1.fedchain:8051
            ;;
        3)
            export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/crypto/peerOrganizations/org1.fedchain/peers/peer3.org1.fedchain/tls/ca.crt
            export CORE_PEER_ADDRESS=peer3.org1.fedchain:9051
            ;;
        *)
            exit 1
    esac
}

echoEnvVar()
{
    echo CORE_PEER_TLS_ENABLED=$CORE_PEER_TLS_ENABLED
    echo CORE_PEER_LOCALMSPID=$CORE_PEER_LOCALMSPID
    echo CORE_PEER_TLS_ROOTCERT_FILE=$CORE_PEER_TLS_ROOTCERT_FILE
    echo CORE_PEER_MSPCONFIGPATH=$CORE_PEER_MSPCONFIGPATH
    echo CORE_PEER_ADDRESS=$CORE_PEER_ADDRESS
}

if [ $# -ne 1 ]; then
    return 1
else
    setEnvVar $1
    echoEnvVar
fi
